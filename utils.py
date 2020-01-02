"""Text-processing utils and other helper functions"""

import torch


# ----------------------------------------------------------------------------
# Text Preprocess
# ----------------------------------------------------------------------------

def get_char_ngrams(word, len_from=3, len_to=6):
    ngrams = []
    for size in range(len_from, len_to+1):
        if len(word) < size:
            break
        i = 0
        while i + size <= len(word):
            ngrams.append(word[i:i+size])
            i += 1
    return ngrams


# ----------------------------------------------------------------------------
# Monitoring
# ----------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ----------------------------------------------------------------------------
# Other tools
# ----------------------------------------------------------------------------


def get_cluster_members(cluster_idx, seq, cutoffs, ent_slices):
    """Given a cluster index, cutoffs, and ent_slices, this computes the mask
    and relative indices in a cluster over the sequence. Mask over the sequence 
    indicates whether it is a member item of the cluster. Relative indices tells
    the positions in the cluster.
    We assume regular cluster comes before the entity clusters in the cluster 
    list. 
    For example, with the settings of cutoffs = [50, 1000] and
    ent_slices=[(100, 200)], the cluster list becomes [R1, R2, R3, E1] where
    the constituent segments are as below:
    - R1: [0, 50)
    - R2: [50, 100), [200, 1100)
    - R3: [1100, N)
    - E1: [100, 200)

    This identifies all the segments for the given cluster, and returns the mask
    and relative indices over the seq.
    
    :param cluster_idx: Cluster index
    :param seq: Sequence of vocabulary indices for an input text
    :param cutoffs: List of boundary indices
    :param ent_slices: Slices for entity sets
    :return: mask and relative indices
    """
    # Offsets
    offsets = []  # pair in (next cutoff, cumulated offset)
    n_entities = 0
    for slice_ in ent_slices:
        offsets.append((slice_.start - n_entities, n_entities))
        n_entities += slice_.stop - slice_.start
    offsets.append((cutoffs[-1], n_entities))

    # Cluster slices
    cluster_slices = []
    cutoff_val = [0] + cutoffs
    if cluster_idx < len(cutoffs):  # Clusters for regular words
        low, hi = cutoff_val[cluster_idx], cutoff_val[cluster_idx+1]
        agg = 0
        for boundary, offset in offsets:
            if low > boundary:
                continue
            if hi - low - agg < boundary:
                cluster_slices.append(slice(low+offset+agg, hi+offset))
                break
            else:
                cluster_slices.append(slice(low+offset+agg, boundary+offset))
                agg = boundary - low
    else:
        cluster_slices.append(ent_slices[cluster_idx-len(cutoffs)])

    # Mask and Relative indices of current cluster over seq
    target_mask = seq.new_zeros(seq.size(), dtype=torch.bool)
    rel_inds = seq.new_full(seq.size(), -1)
    rel_offset = 0
    for slice_ in cluster_slices:
        low, high = slice_.start, slice_.stop
        mask = (seq >= low) & (seq < high)
        rel_inds.index_copy_(0, mask.nonzero().squeeze(),
                             seq[mask] - low + rel_offset)
        target_mask |= mask.bool()
        rel_offset += high - low
    return target_mask, rel_inds


# Rank aggregation methods
def rank_agg_borda(rank_lists):
    raise NotImplementedError


def rank_agg_rrf(score_lists):
    k = 60
    scores = [0] * len(score_lists[0])
    for list in score_lists:
        assert len(list) == len(scores), "Length of a score list does not match"
        ranks = [sorted(list, reverse=True).index(x)+1 for x in list]
        for i, r in enumerate(ranks):
            scores[i] += 1 / (k + r)
    return scores
