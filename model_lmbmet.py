"""Model definitions of LmBMET, which has the Transformer-XL language model for
 training LmBMET embeddings"""
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from crit import BMETAdaptiveLogSoftmaxWithLoss
from utils import get_cluster_members

logger = logging.getLogger(__name__)


class Highway(nn.Module):
    """Highway network to gate between linear/non-linear outputs"""
    def __init__(self, size, num_layers=2, act_fn=F.selu):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = \
            nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = \
            nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = \
            nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.act_fn = act_fn

    def forward(self, x):
        """
        :param x: tensor in [batch_size x size]
        :return: tensor in [batch_size x size]

        s(x) @ (f(G(x)) + (1 - s(x)) @ Q(x)
        G and Q are affine transformation
        f is a non-linear transformation
        s is a sigmoid gate
        and @ is element-wise multiplication
        """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.act_fn(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        # Store inv_freq in state_dict, but not in parameters
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # outer product
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_heads, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head

        # Multi-head: QKV (3) * d_model (n_heads * d_head)
        self.qkv_net = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        # Dropout applied over the attention probabilities
        self.dropatt = nn.Dropout(dropatt)
        # Linear projection
        self.o_net = nn.Linear(n_heads * d_head, d_model, bias=False)
        # Dropout after linear projection
        self.drop = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        # "Scaled dot-product"
        self.scale = 1 / (d_head ** 0.5)

        # Layer normalization on inputs
        self.pre_lnorm = pre_lnorm

    @staticmethod
    def _parallelogram_mask(h, w, left=False):
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    @staticmethod
    def _shift(x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # To be overridden
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        # FF in transformer encoder
        self.r_net = nn.Linear(self.d_model, self.n_heads * self.d_head,
                               bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_heads, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_heads, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_heads, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_heads, self.d_head)  # W_{k,R} R

        # compute attention score
        rw_head_q = w_head_q + r_w_bias  # E_{x_i}' W_q' + u'
        # einsum: Einstein notation (http://ajcr.net/Basic-guide-to-einsum/)
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias  # E_{x_i}' W_q' + v'
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)  # Masked-Softmax
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector (a_\tau^n)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_heads * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, args, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            args.n_heads, args.d_model, args.d_head, args.dropout, **kwargs)
        self.pos_ff = PositionwiseFF(args.d_model, args.d_inner, args.dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, inp, r, r_w_bias, r_r_bias, dec_attn_mask=None,
                mems=None):
        output = self.dec_attn(inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class AdaptiveEmbedding(nn.Module):
    """Adaptive input representations by Baevski et al. [2018]"""
    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1.,
                 ent_slices=None, ent_hsz=None):
        super(AdaptiveEmbedding, self).__init__()

        # Bookkeeping
        self.n_tokens = vocab_size                  # Vocabulary size
        self.d_embed = d_embed                      # Embedding dimension
        cutoffs = [] if cutoffs is None else list(cutoffs)
        assert vocab_size > cutoffs[-1], \
            "Vocabulary size is less than the last item of the Adaptive " \
            "embedding cutoffs"
        self.cutoffs = cutoffs + [vocab_size]       # Cutoff boundaries
        self.div_val = div_val                      # Division factor, k
        self.d_proj = d_proj                        # Output dimension
        self.emb_scale = d_proj ** 0.5              # Scale factor at the end
        self.ent_slices = [] if ent_slices is None else ent_slices
                                                    # slices of entity sets
        self.ent_hsz = [] if ent_hsz is None else ent_hsz
                                                    # hidden dimensions for ent

        # We do not apply adaptive embedding on entity codes. Hence, we need to
        # know where entity codes are listed in the vocabulary and how many are
        # those
        self.offsets = []
        if len(ent_slices) > 0:
            assert len(ent_slices) == len(ent_hsz), \
                "The size of ent_hsz should match the number of entity sets"
            n_ent = []
            for slice_ in ent_slices:
                self.offsets.append((slice_.start - sum(n_ent), sum(n_ent)))
                n_ent.append(slice_.stop - slice_.start)
            self.offsets.append((vocab_size - sum(n_ent), sum(n_ent)))
            self.n_entities = n_ent
        else:
            self.offsets.append((vocab_size, 0))

        self.n_clusters = len(self.cutoffs) + len(ent_slices)
        # Embedding layers of different clusters
        self.emb_layers = nn.ModuleList()
        # Projection layers of different clusters
        self.emb_projs = nn.ParameterList()

        if div_val == 1:  # i.e., regular embedding and projection layers
            self.emb_layers.append(nn.Embedding(vocab_size, d_embed))
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed))
                )
        else:
            cutoff_val = [0] + self.cutoffs
            for i in range(self.n_clusters):
                if i < len(self.cutoffs):  # Regular words
                    cluster_size = cutoff_val[i+1] - cutoff_val[i]
                    d_emb_i = d_embed // (div_val ** i)
                else:  # Entity codes
                    idx = i - len(self.cutoffs) - 1
                    cluster_size = ent_slices[idx].stop - ent_slices[idx].start
                    d_emb_i = ent_hsz[idx]
                self.emb_layers.append(nn.Embedding(cluster_size, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

    def forward(self, inp):
        param = next(self.parameters())
        if isinstance(inp, list):
            inp = torch.LongTensor(inp).to(param.device)
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            inp_flat = inp.flatten()
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs) + len(self.ent_slices)):
                target_mask, rel_inds = \
                    get_cluster_members(i, inp_flat, self.cutoffs, self.ent_slices)
                row_indices = target_mask.nonzero().squeeze()
                if row_indices.numel() == 0:
                    continue
                inp_i = rel_inds.index_select(0, row_indices)
                # Embedding
                emb_i = self.emb_layers[i](inp_i)
                # Projection
                emb_i = F.linear(emb_i, self.emb_projs[i])
                # Copy
                emb_flat.index_copy_(0, row_indices, emb_i)
            # Restore seq/bsz dimensions
            embed = emb_flat.view(*inp.size(), self.d_proj)
        # Scale
        embed.mul_(self.emb_scale)

        return embed


class LmBMET(nn.Module):
    """LmBMET using MemTransformerLM model class"""
    def __init__(self, args, vocab_size=None, cutoffs=None,
                 ent_slices=None, ent_hsz=None):
        super(LmBMET, self).__init__()

        # Book-keeping
        self.args = args
        self.d_embed = args.d_embed
        self.d_model = args.d_model
        self.vocab_size = vocab_size   # Vocabulary size
        self.n_layers = args.n_layers  # Num of transformer layers
        self.n_heads = args.n_heads    # In multi-head attentions
        self.d_head = args.d_head      # head dimension
        self.tgt_len = args.tgt_len    # Num of tokens to predict
        self.mem_len = args.mem_len    # Retained previous heads
        self.ext_len = args.ext_len    # Length of extended context
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len
        self.ent_slices = [] if ent_slices is None else ent_slices
        self.ent_hsz = [] if ent_hsz is None else ent_hsz
        self.same_length = args.same_length
        self.clamp_len = args.clamp_len

        # Create parameters
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))

        # Embeddings
        self.pos_emb = PositionalEmbedding(self.d_model)
        if args.wbmet_file != '':
            self.word_emb = nn.Embedding(self.vocab_size, args.d_embed)
            self.word_emb.weight.requires_grad = False
            self.emb_proj = nn.Parameter(torch.Tensor(args.d_model, args.d_embed))
        else:
            self.word_emb = AdaptiveEmbedding(
                vocab_size, self.d_embed, self.d_model, cutoffs,
                div_val=args.div_val, ent_slices=self.ent_slices,
                ent_hsz=ent_hsz
            )

        # Dropout
        self.drop = nn.Dropout(args.dropout)

        # Transformer layers (Dai's default attention type)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                RelPartialLearnableDecoderLayer(self.args, pre_lnorm=True)
            )

        # Loss
        self.div_val = args.div_val  # Divident value of adaptiveEmbedding

        # Use Adaptive Softmax (including standard softmax)
        self.crit = BMETAdaptiveLogSoftmaxWithLoss(
            self.d_model, self.vocab_size, cutoffs, div_value=args.div_val,
            ent_slices=self.ent_slices, ent_hsz=self.ent_hsz
        )

    def read_emb_from_pretrained(self, vocab):
        logger.info('Reading Embedding vectors from wbmet file')
        weight = self.word_emb.weight
        with self.args.wbmet_file.open() as f:
            _, dim = map(int, next(f).split()[:2])
            cnt = 0
            for line in f:
                vals = line.split()
                try:
                    weight[vocab.sym2idx[vals[0]]] = \
                        torch.FloatTensor(list(map(float, vals[1:dim+1])))
                except KeyError:
                    continue
                else:
                    cnt += 1
                if cnt % 1000 == 0:
                    print('{} - {:20}\r'.format(cnt, vals[0]), end='')
        logger.info('{} wbmet vectors copied'.format(cnt))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        
    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layers + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, data, target, *mems):
        # First pass
        if not mems:
            mems = self.init_mems()

        # Set dimensions
        tgt_len = target.size(0)
        mlen = mems[0].size(0) if mems is not None else 0
        qlen, bsz = data.size()[:2]
        klen = mlen + qlen

        # Token representations
        word_emb = self.word_emb(data)
        if self.args.wbmet_file != '':
            word_emb = F.linear(word_emb, self.emb_proj)

        # Attention mask
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                     + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None]
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen),
                                       diagonal=1+mlen).bool()[:, :, None]

        # Positional embeddings
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:  # Use the same pos embeddings after clamp_len
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        # Hidden outputs of the transformer layers
        hids = [core_out]  # n_layer x N x eval_tgt_len x d_model
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                             dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        hidden = self.drop(core_out)  # N x eval_tgt_len x d_model
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        # Predict and get Loss
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss], hids, pred_hid
        else:
            return [loss] + new_mems, hids, pred_hid
