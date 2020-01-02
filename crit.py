"""Criterion layers"""

import pprint
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_cluster_members

logger = logging.getLogger(__name__)


class BMETAdaptiveLogSoftmaxWithLoss(nn.Module):
    """Adaptive Softmax with dedicated entity code clusters
    (Refer to "Efficient softmax approximation" by Grave et al.)

    * :attr:`cutoffs` is an ordered sequence of integers indicating the
      boundaries of clusters.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.
    * :attr:`offsets` sequence of offsets that is a pair of boundary index and
      its offset which to be added to the cutoff slices. For example, an offset
      `(i, d)` tells that elements up to the relative index `i` should add the
      number of entities `d` to the index.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset, i.e., vocabulary size
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value: Determines the projection layer dimension of each cluster
        head_bias: Add bias to the head cluster layer
        ent_slices: list of slices which indicate the entity codes in vacabulary
        ent_hsz: the projection layer dimensions for the entity sets.

    Shape:
        - input: [N, d_proj], where N = len * bsz
        - target: [N]
        - output1: [N]
        - output2: Scalar

    (Refer to the PyTorch implementation torch.nn.AdaptiveLogSoftmaxWithLoss)
    """
    def __init__(self, in_features, n_classes, cutoffs, div_value=2.,
                 ent_slices=None, ent_hsz=None, head_bias=False):
        super(BMETAdaptiveLogSoftmaxWithLoss, self).__init__()

        self.offsets = []
        self.n_entities = 0
        if ent_slices is not None:
            assert len(ent_slices) == len(ent_hsz), \
                "The size of ent_hsz should match the number of entity sets"
            for slice_ in ent_slices:
                self.offsets.append((slice_.start - self.n_entities,
                                     self.n_entities))
                self.n_entities += slice_.stop - slice_.start
            # Add the remaining offset
            self.offsets.append((n_classes - self.n_entities, self.n_entities))
        else:
            self.offsets.append((n_classes, 0))
            ent_slices = []
            ent_hsz = []

        end = n_classes - self.n_entities
        self.cutoffs = [end] if cutoffs is None else list(cutoffs) + [end]
        self.in_features = in_features
        self.n_classes = n_classes
        self.div_value = div_value
        self.head_bias = head_bias
        self.ent_slices = ent_slices
        self.n_clusters = len(self.ent_slices) + len(self.cutoffs) - 1
        self.head_size = self.cutoffs[0] + self.n_clusters

        self.head = nn.Linear(self.in_features, self.head_size,
                              bias=self.head_bias)
        self.tail = nn.ModuleList()

        # Add projection layers for the regular-word clusters
        for i in range(self.n_clusters):
            if i < len(cutoffs):  # Regular words
                hsz = int(self.in_features // (self.div_value ** (i + 1)))
                osz = self.cutoffs[i + 1] - self.cutoffs[i]
            else:  # Entity codes
                hsz = ent_hsz[i-len(cutoffs)]
                slice_ = ent_slices[i-len(cutoffs)]
                osz = slice_.stop - slice_.start
            projection = nn.Sequential(
                nn.Linear(self.in_features, hsz, bias=False),
                nn.Linear(hsz, osz, bias=False)
            )
            self.tail.append(projection)

    def reset_parameters(self):
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, inp: torch.FloatTensor, tgt: torch.LongTensor):
        if inp.size(0) != tgt.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')
        num_elms = 0
        entry_size = tgt.size(0)
        output = inp.new_zeros(entry_size)        # log probabilities
        gather_inds = tgt.new_empty(entry_size)   # tgt indices in head

        for i in range(self.n_clusters + 1):
            target_mask, rel_inds = \
                get_cluster_members(i, tgt, self.cutoffs, self.ent_slices)
            # members of the current cluster
            members = target_mask.nonzero().squeeze()
            if members.numel() == 0:
                continue
            if i == 0:  # Head cluster
                # Head cluster also needs to compute relative indices
                gather_inds.index_copy_(0, members, rel_inds[target_mask])
            else:  # Tail clusters including entity clusters
                cluster_index = self.cutoffs[0] + i - 1
                gather_inds.index_fill_(0, members, cluster_index)

                # Subset of input which elements should be in this cluster
                input_subset = inp.index_select(0, members)
                # Forward
                cluster_output = self.tail[i - 1](input_subset)
                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                relative_target = rel_inds[target_mask]
                local_logprob = \
                    cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, members, local_logprob.squeeze(1))

            num_elms += members.numel()

        if num_elms != entry_size:
            logger.error('used_rows ({}) and batch_size ({}) does not match'
                         ''.format(num_elms, entry_size))
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     tgt.min().item(),
                                                     tgt.max().item()))

        head_output = self.head(inp)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()

        # return neglog
        return -output

    def _get_full_log_prob(self, inp, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = inp.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        for col in range(self.head_size):
            from_ = 0
            for offset in self.offsets:
                if from_ <= col < offset[0]:
                    out[:, col+offset[1]] = head_logprob[:, col]
                from_ = offset[0]

        # Regular words
        for i, (start_idx, stop_idx) in \
                enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](inp)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + \
                             head_logprob[:, self.cutoffs[0] + i].unsqueeze(1)

            for col in range(start_idx, stop_idx):
                from_ = 0
                for offset in self.offsets:
                    if from_ <= col < offset[0]:
                        out[:, col+offset[1]] = output_logprob[:, col-start_idx]
                    from_ = offset[0]

        # Entities
        for i, slice_ in enumerate(self.ent_slices):
            i_ = i + len(self.cutoffs) - 1  # index in the list of all clusters
            cluster_output = self.tail[i_](inp)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + \
                             head_logprob[:, self.cutoffs[0] + i_].unsqueeze(1)

            out[:, slice_.start:slice_.stop] = output_logprob
        return out

    def log_prob(self, inp):
        r""" Computes log probabilities for all :math:`n\_classes`

        Args:
            inp (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= n\_classes`, where :math:`n\_classes` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, in\_features)`
            - Output: :math:`(N, n\_classes)`

        """

        head_output = self.head(inp)
        return self._get_full_log_prob(inp, head_output)

    def predict(self, inp):
        r""" This is equivalent to `self.log_pob(input).argmax(dim=1)`,
        but is more efficient in some cases.

        Args:
            inp (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, in\_features)`
            - Output: :math:`(N)`
        """
        head_output = self.head(inp)
        output = torch.argmax(head_output, dim=2)
        not_in_shortlist = (output >= self.cutoffs[0])
        all_in_shortlist = not (not_in_shortlist.any())

        def remap_to_raw_index(offsets, output):
            from_ = 0
            for offset in offsets:
                output[(from_ <= output) & (output < offset[0])] += offset[1]
                from_ = offset[0]
            return output

        if all_in_shortlist:
            return remap_to_raw_index(self.offsets, output)

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(inp, head_output)
            return remap_to_raw_index(self.offsets, torch.argmax(log_prob, dim=1))

        else:
            log_prob = self._get_full_log_prob(inp[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return remap_to_raw_index(self.offsets, output)


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    opt = {
        'in_features': 50,
        'n_classes': 300,
        'cutoffs': [5, 50, 100],
        'ent_slices': [slice(2, 80)],
        'ent_hsz': [10],
        'div_value': 2.
    }
    # pp.pprint(opt)

    len_seq = 10
    bsz = 1

    crit = BMETAdaptiveLogSoftmaxWithLoss(**opt)
    target = torch.LongTensor([[0,1,5,80,84,128,180,72,0,3]])
    print('target', target)
    hdn = torch.randn(bsz, len_seq, opt['in_features'])
    nll = crit.forward(hdn.view(-1, hdn.size(-1)), target.view(-1))
    print('output {}'.format(nll))
    print('loss {}'.format(nll.mean()))
