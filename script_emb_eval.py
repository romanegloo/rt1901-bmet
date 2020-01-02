#!/usr/bin/env python3
"""Evaluation script for embedings; word, mesh, and contextualized
representations. This script is used in two modes; validation mode in training
and test mode in final test."""
import argparse
import logging
import csv
from collections import defaultdict
import pickle

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors
import torch
import numpy as np
from tabulate import tabulate
from nltk.tokenize import word_tokenize

from utils import rank_agg_rrf
logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    eval_files = {
        'UMNSRS-sim': ('UMNSRS_sim_mesh.csv', (2, 3, 6, 7, 0)),
        'UMNSRS-rel': ('UMNSRS_rel_mesh.csv', (2, 3, 6, 7, 0)),
        'UMNSRS-sim-mod': ('UMNSRS_sim_mod_mesh.csv', (2, 3, 6, 7, 0)),
        'UMNSRS-rel-mod': ('UMNSRS_rel_mod_mesh.csv', (2, 3, 6, 7, 0)),
        'MayoSRS': ('MayoSRS_mesh.csv', (3, 4, 5, 6, 0)),
        'MiniMayoSRS-p': ('MiniMayoSRS_mesh.csv', (4, 5, 6, 7, 0)),
        'MiniMayoSRS-c': ('MiniMayoSRS_mesh.csv', (4, 5, 6, 7, 1)),
        'Pedersen-p': ('Pedersen2007_mesh.csv', (0, 1, 2, 3, 4)),
        'Pedersen-c': ('Pedersen2007_mesh.csv', (0, 1, 2, 3, 5)),
        'Hliaoutakis': ('Hliaoutakis2005_mesh.csv', (0, 1, 2, 3, 4))
    }

    def __init__(self, eval_sets, model=None, vocab=None, mesh_def=None):
        self.mode = 'validate'
        self.eval_wv_only = False
        self.dir_eval = 'data/eval/'
        self.eval_sets = eval_sets
        self.mesh_indicator = 'εmesh_'
        self.mesh_indicator_desc = self.mesh_indicator + 'd'
        self.eval_vocab = None
        self.vocab_embs = None
        self.wv_bin_mdl = None
        self.mdl = model
        # If CUDA is available, load on GPU. Otherwise, on CPU
        self.mdl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = vocab
        self.mesh_def = mesh_def
        self.scoreboard = None
        self.sc_models = None  # list of prediction models
        self.freq_stat = {}

        # Read evaluation examples into dict
        self.reference = self.read_eval_sets()

    def read_eval_sets(self):
        """ Read evaluation datasets and store them in a unified format. All the
        embeddings for the vocabulary is cached in *vocab_embs*."""
        # reference records in [t1, t2, m1, m2, score1]
        ref = defaultdict(list)
        self.eval_vocab = set()
        for ds, (filename, mapping) in self.eval_files.items():
            with open(self.dir_eval + filename) as f:
                csv_reader = csv.reader(map(lambda line: line.lower(), f))
                next(csv_reader)  # Skip the header row
                for rec in csv_reader:
                    ex = [None] * 5
                    for i, v in enumerate(mapping):
                        if i < 2:  # terms
                            ex[i] = rec[mapping[i]]
                            self.eval_vocab.update(ex[i].split())
                        elif i < 4:  # mesh terms
                            if rec[mapping[i]] != 'none':
                                ex[i] = self.mesh_indicator + rec[mapping[i]]
                                self.eval_vocab.add(ex[i])
                            else:
                                ex[i] = 'none'
                        else:  # score values
                            ex[i] = float(rec[mapping[i]])
                    ref[ds].append(ex)
        self.vocab_embs = {k: [None, None] for k in self.eval_vocab}
        self.vocab_embs['<unk>'] = [None, None]
        return ref

    def load_from_file(self, wv_file, lmbmet_file, corpus_file):
        logger.info('Reading word embeddings...')
        self.eval_wv_only = (None in [args.lmbmet_file, args.corpus_file])
        if wv_file.endswith('.bin'):
            wv_bin_mdl = KeyedVectors.load_word2vec_format(wv_file, binary=True)
            for k in self.vocab_embs.keys():
                try:
                    self.vocab_embs[k][0] = wv_bin_mdl[k].tolist()
                except KeyError as e:
                    continue
        else:
            with open(wv_file) as f:
                next(f)
                for i, line in enumerate(f):
                    vals = line.split()
                    if vals[0] in self.vocab_embs:
                        self.vocab_embs[vals[0]][0] = list(map(float, vals[1:]))
                        self.freq_stat[vals[0]] = i
                logger.info('vocabulary size {}'.format(i))

        if self.eval_wv_only:
            return
        logger.info('Loading corpus...')
        corpus = pickle.load(open(corpus_file, 'rb'))
        v = corpus.vocab

        logger.info('Loading saved LMBET model...')
        self.mdl = torch.load(lmbmet_file, map_location=self.mdl_device)

        # Load context-independent embeddings for eval_vocab in LMBMET model
        vocab_tensors = \
            v.convert_to_tensor(self.vocab_embs.keys()).to(self.mdl_device)
        embeddings = self.mdl.word_emb(vocab_tensors)
        assert embeddings.shape[0] == len(self.vocab_embs), \
            "Not all vocabs are converted into lmbmet embeddings"
        for i, vec in zip(vocab_tensors, embeddings):
            self.vocab_embs[v.idx2sym[i]][1] = vec.tolist()

        # Load context-dependent embeddings for mesh terms in LMBMET model
        for ent in tqdm(self.vocab_embs.keys(), desc='mesh-def'):
            if not ent.startswith(self.mesh_indicator_desc):
                continue
            if ent not in corpus.mesh_def:
                continue
            def_ = corpus.mesh_def[ent]
            context = def_['note'] + ' ' + def_['name'] + ' ' + ent
            title_len = len(word_tokenize(def_['name'] + ' ' + ent))
            context = word_tokenize(context.lower())
            tensors = v.convert_to_tensor(context)
            tensors = torch.unsqueeze(tensors, dim=0).to(self.mdl_device)
            _, hids, pred_hid = self.mdl(tensors, tensors)
            for l in range(len(hids)):
                emb = hids[l][0][-title_len:].mean(dim=0).tolist()
                self.vocab_embs[ent].append(emb)

    def cosine_sim(self, term1, term2):
        # We have three types of cosine similarity scores; (1) word-level sim.
        # (2) lmbmet embedding layer sim., and (3) contextualized mesh sim.
        score_dim = 2
        if not self.eval_wv_only:
            score_dim += self.mdl.n_layers + 1  # +1: input emb
        embs = [[None, None] for _ in range(score_dim)]
        scores = [None] * score_dim
        if 'none' in [term1, term2]:
            return scores

        for i, t in enumerate([term1, term2]):
            # 1. word-level sim. using wv_model
            if self.mode == 'test':
                vecs = [self.vocab_embs[w][0] for w in t.lower().split()
                        if self.vocab_embs[w][0] is not None]
                if len(vecs) > 0:
                    embs[0][i] = np.mean(np.array(vecs), axis=0)
            # 2. lmbmet embedding layer sim.
            if self.mode == 'test':
                vecs = [self.vocab_embs[w][1] for w in t.lower().split()
                        if self.vocab_embs[w][1] is not None]
                if len(vecs) > 0:
                    embs[1][i] = np.mean(np.array(vecs), axis=0)
            else:  # validate mode
                idx = []
                for w in t.lower().split():
                    try:
                        idx.append(self.vocab.sym2idx[w])
                    except KeyError as e:
                        continue
                if len(idx) > 0:
                    inp = torch.LongTensor(idx).to(self.mdl_device)
                    tensors = self.mdl.word_emb(inp)
                    # embs[i].append(tensors.mean(dim=0).detach().cpu().numpy())
                    embs[1][i] = tensors.mean(dim=0).detach().cpu().numpy()
            # (3) contextualized mesh sim.
            if self.eval_wv_only or not t.startswith(self.mesh_indicator_desc):
                continue
            if self.mode == 'test':
                for l in range(2, score_dim):
                    embs[l][i] = self.vocab_embs[t][l]
            else:  # validate mode
                def_ = self.mesh_def[t]
                context = def_['note'] + ' ' + def_['name'] + ' ' + t
                title_len = len(word_tokenize(def_['name'] + ' ' + t))
                context = word_tokenize(context.lower())
                tensors = self.vocab.convert_to_tensor(context)
                tensors = torch.unsqueeze(tensors, dim=0).to(self.mdl_device)
                _, hids, pred_hid = self.mdl(tensors, tensors)
                for l in range(len(hids)):
                    embs[l+2][i] = hids[l][0][-title_len:].mean(dim=0).tolist()

        for i, v in enumerate(embs):
            if any(elm is None for elm in v):
                if not term1.startswith(self.mesh_indicator_desc):
                    scores[i] = np.random.beta(2, 2)
                continue
            scores[i] = 1 - cosine(embs[i][0], embs[i][1])

        return scores

    def eval(self):
        if self.mode == 'validate':
            self.sc_models = ['lm-term', 'lm-mesh']
            layers = ['lm-mesh-def-' + str(i)
                      for i in range(self.mdl.n_layers + 1)]
            self.sc_models += layers
            self.sc_models.append('rank-fusion')
        else:  # test mode
            if self.eval_wv_only:
                self.sc_models = ['w-term', 'w-mesh']
            else:
                self.sc_models = ['w-term', 'w-mesh', 'lm-term', 'lm-mesh']
                layers = ['lm-mesh-def-' + str(i)
                          for i in range(self.mdl.n_layers + 1)]
                self.sc_models += layers
                self.sc_models.append('rank-fusion')
        self.scoreboard = np.zeros((len(self.sc_models), len(self.eval_sets)))

        # Evaluate by sets
        for idx, ds in enumerate(self.eval_sets):
            scores = defaultdict(list)
            for ex in self.reference[ds]:
                scores['gt'].append(ex[4])  # gt score
                # Using terms
                sims = self.cosine_sim(*ex[:2])
                w_t, lm_t, hids_t = sims[0], sims[1], sims[2:]
                if 'w-term' in self.sc_models:
                    scores['w-term'].append(w_t)
                if 'lm-term' in self.sc_models:
                    scores['lm-term'].append(lm_t)

                # Using meshes
                sims = self.cosine_sim(*ex[2:4])
                w_e, lm_e, hids_e = *sims[:2], sims[2:]

                # Fallback to term-based similarity, if mesh is OOV
                if 'w-mesh' in self.sc_models:
                    scores['w-mesh'].append(w_t if w_e is None else w_e)
                if 'lm-mesh' in self.sc_models:
                    scores['lm-mesh'].append(w_t if lm_e is None else lm_e)
                if not self.eval_wv_only:
                    # This returns the scores of each hidden outputs of n layers
                    for i in range(self.mdl.n_layers + 1):
                        k = 'lm-mesh-def-' + str(i)
                        try:
                            scores[k].append(w_t if hids_e[i] is None else hids_e[i])
                        except IndexError as e:
                            print(k, len(hids_e), i)
                            raise

            # Add rank-fusion scores
            if 'rank-fusion' in self.sc_models:
                if self.mode == 'validate':
                    fusion_scores = ['lm-term', 'lm-mesh', 'lm-mesh-def-0',
                                     'lm-mesh-def-12']
                else:
                    fusion_scores = ['w-term', 'w-mesh',
                                     'lm-mesh-def-1', 'lm-mesh-def-12']
                # RRF
                scores['rank-fusion'] = \
                    rank_agg_rrf([scores[k] for k in fusion_scores])
                # combSUM
                # scores['rank-fusion'] = [0] * len(scores['gt'])
                # for k in fusion_scores:
                #     scores['rank-fusion'] = list(map(add, scores['rank-fusion'],
                #                                      rankdata(scores[k])))

            # Update scoreboard
            for k in [k for k in scores.keys() if k != 'gt']:
                rho, p = spearmanr(scores['gt'], scores[k])
                self.scoreboard[self.sc_models.index(k)][idx] = rho

        total_score = 0
        for k in ['lm-term', 'lm-mesh', 'lm-mesh-def-0', 'lm-mesh-def-12']:
            if k in self.sc_models:
                total_score += self.scoreboard[self.sc_models.index(k)][-5:].mean()

        # Print out the scoreboard
        self.scoreboard = np.hstack(
            (np.expand_dims(np.array(self.sc_models), axis=1),
             self.scoreboard)
        )
        tbl = tabulate(
            self.scoreboard, headers=['model'] + self.eval_sets, tablefmt='rst'
        )
        logger.info('\n{}'.format(tbl))

        return total_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wv_file', type=str, default=None,
                        help='Path to a pre-trained word embeddings file '
                             'or wbmet file')
    parser.add_argument('--lmbmet_file', type=str, default=None,
                        help='Path to a trained lmbmet PyTorch model file')
    parser.add_argument('--corpus_file', type=str, default=None,
                        help='Path to the encoded PubTator corpus file')
    args = parser.parse_args()

    # Logger
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Set evaluation datasets
    eval_sets = list(EmbeddingEvaluator.eval_files.keys())
    eval_sets = ['UMNSRS-sim', 'UMNSRS-rel', 'MayoSRS',
                 'MiniMayoSRS-p', 'MiniMayoSRS-c',
                 'Pedersen-p', 'Pedersen-c', 'Hliaoutakis']

    # Load evaluator
    evaluator = EmbeddingEvaluator(eval_sets=eval_sets)
    evaluator.load_from_file(args.wv_file, args.lmbmet_file, args.corpus_file)
    evaluator.mode = 'test'

    # Print Frequency statistics by dataset
    for ds in eval_sets:
        words = []
        meshes = []
        for rec in evaluator.reference[ds]:
            for i in range(2):
                for t in rec[i].split():
                    if t in evaluator.freq_stat:
                        words.append(evaluator.freq_stat[t])
                    else:
                        words.append(-1)
            for i in range(2, 4):
                for t in rec[i].split():
                    if t in evaluator.freq_stat:
                        meshes.append(evaluator.freq_stat[t])
                    else:
                        meshes.append(-1)
        print('dataset:', ds)
        wp = [v for v in words if v > 0]
        wm = [v for v in meshes if v > 0]
        print('- coverage (word, mesh): {:.2f} {:.2f}'
              .format(len(wp) / len(words), len(wm) / len(meshes)))
        print('- mean freq. rank (word, mesh): {:.2f} {:.2f}'
              .format(sum(wp)/len(wp), sum(wm)/len(wm)))



    # Run tests
    logger.info('-'*40 + ' Running Evaluation ' + '-'*40)
    score = evaluator.eval()
    logger.info('Evaluation completed\n' + '-'*100)
