"""Data structures needed for preparing training datasets which include

- Vocab and Corpus class
- PyTorch Dataset, DataLoader, and Sampler
"""
import pickle
import logging
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Vocab and Corpus
# ------------------------------------------------------------------------------
class Vocab:
    """Vocabulary in word-level symbols and auxilary tools"""
    def __init__(self):
        self.idx2sym = list()
        self.sym2idx = OrderedDict()
        self.sizes = [0] * 3

    def load_vocab(self, words, mesh_def=None, specials=None):
        if specials is not None:
            for sym in specials:
                self.add_special(sym)
                self.sizes[0] += 1
        if mesh_def is not None:
            for k, data in mesh_def.items():
                if data['descriptor']:
                    self.add_symbol(k)
                    self.sizes[1] += 1
        for w in words:
            self.add_symbol(w)
            self.sizes[2] += 1

    def convert_to_tensor(self, symbols):
        """Converts input sequences into a tensor of symbol indices
        return LongTensor of symbol indices: [num_seq x 1]
        """
        tensors = [self.get_idx(sym) for sym in symbols]
        return torch.LongTensor(tensors)

    def get_idx(self, sym):
        return self.sym2idx.get(sym, self.unk_idx)

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def encode_symbols(self, inp):
        if isinstance(inp, str):
            symbols = inp.split()
            if len(symbols) == 1:
                return self.sym2idx[inp] if inp in self.sym2idx else self.unk_idx
        else:
            symbols = inp
        return [self.sym2idx[sym] if sym in self.sym2idx else self.unk_idx
                for sym in symbols]

    def decode_symbols(self, symbols):
        try:
            text = ' '.join([self.idx2sym[idx] for idx in symbols])
        except IndexError as e:
            logger.error('symbols: {}'.format(symbols))
            logger.error('vocab_size {}'.format(len(self)))
            raise e
        return text

    def __len__(self):
        return len(self.idx2sym)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.idx2sym[item]
        elif isinstance(item, str):
            return self.sym2idx[item]
        else:
            raise IndexError


class Corpus:
    """Read corresponding data files, encode data"""
    def __init__(self):
        self.vocab = Vocab()
        self.mesh_def = None
        self.train = self.valid = self.test = None

    def load_data(self, docs):
        """Read corpus in"""
        logger.info('>> Reading and encoding the corpus...')
        # Encode docs
        seq = []
        for doc in docs:
            seq.extend(self.vocab.encode_symbols(doc))

        # Shuffle to mix PubTator docs and distribute them into datasets
        # train(80%), valid(10%), test(10%). Append mesh definitions to the
        # train set
        chunk_sz = 50000
        self.train = []
        self.valid = []
        self.test = []
        for i in range(0, len(seq), chunk_sz):
            if (i / chunk_sz) % 10 >= 9:
                self.test.append(torch.LongTensor(seq[i:i+chunk_sz]))
            elif (i / chunk_sz) % 10 >= 8:
                self.valid.append(torch.LongTensor(seq[i:i+chunk_sz]))
            else:
                self.train.append(torch.LongTensor(seq[i:i+chunk_sz]))

        # Encode mesh defs
        logger.info('>>>> adding MeSH definition to the end of train set')
        cnt = 0
        train_prepend = []
        if self.mesh_def is not None:
            for k in self.mesh_def:
                if 'enc_body' in self.mesh_def[k]:
                    cnt += 1
                    val = self.vocab.encode_symbols(self.mesh_def[k]['enc_body'])
                    train_prepend.append(torch.LongTensor(val))
                    self.train.append(torch.LongTensor(val))
        logger.info('>>>> {} MeSH definitions added'.format(cnt))

        self.train = torch.cat(train_prepend + self.train)
        self.valid = torch.cat(self.valid)
        self.test = torch.cat(self.test)


# ------------------------------------------------------------------------------
# Dataset and Sampler
# ------------------------------------------------------------------------------
class DatasetLmBMET(Dataset):
    def __init__(self, args, corpus, ds_type=None):
        self.data_src = getattr(corpus, ds_type)
        self.data_tgt = self.data_src

    def __len__(self):
        return self.data_src.size(0)

    def __getitem__(self, idx):
        if len(idx) == 2 and type(idx[0]) == slice and type(idx[1]) == slice:
            return self.data_src[idx[0]], self.data_tgt[idx[1]]
        return self.data_src[idx]


class BPTTBatchSampler(Sampler):
    """Samples data slices sequentially of size bptt_length"""
    def __init__(self, data, batch_size, bptt_len, ext_len):
        self.data_len = len(data)
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.ext_len = ext_len

        # Cleanly, divide data into (batch_size * bptt_len) slices
        self.chunk_size = batch_size * bptt_len
        self.n_batch = self.data_len // self.chunk_size

    def __iter__(self):
        for b in range(self.n_batch):
            src = []
            target = []
            b_start = b * self.bptt_len
            for r in range(self.batch_size):
                r_start = b_start + r * self.chunk_size
                start = max(0, r_start - self.ext_len)
                end = r_start + self.bptt_len
                src.append(slice(start, end))
                target.append(slice(r_start+1, r_start+1+self.bptt_len))

            if len(src) != self.batch_size:
                break

            yield zip(src, target)

    def __len__(self):
        return self.n_batch


def prepare_iterators(args):
    """Reads corpus data files, generate train/val/test datasets,
    return dataloaders and its Vocabulary instance"""
    # Read from cached file
    if args.corpus_file.exists():
        logger.info("Loading Datasets from a cached file...")
        corpus = pickle.load(open(args.corpus_file, 'rb'))
    else:
        raise RuntimeError("Cannot find pickled corpus file")

    iterators = []
    for ds_type in ['train', 'valid', 'test']:
        bsz = args.batch_size if ds_type == 'train' else 8
        ext_len = args.ext_len if args.ext_len is not None else 0
        sampler = BPTTBatchSampler(
            getattr(corpus, ds_type),
            batch_size=bsz,
            bptt_len=args.tgt_len,
            ext_len=ext_len
        )
        iterators.append(
            DataLoader(DatasetLmBMET(args, corpus, ds_type=ds_type),
                       batch_sampler=sampler)
        )
    return corpus, iterators

