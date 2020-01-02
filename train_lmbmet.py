#!/usr/bin/env python3
"""Training character-level and context-aware BMET embeddings on language model
over biomedical corpora. Part of the code is from the original Transformer-XL
implementations"""
import argparse
import logging
import math
import time
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import prepare_iterators
from model_lmbmet import LmBMET
from utils import AverageMeter
from script_emb_eval import EmbeddingEvaluator


###############################################################################
# Configuration
###############################################################################

def add_arguments():
    """Add model parameters to an argparser"""
    # Paths
    parser.add_argument('--data_dir', type=str, default='',
                        help='Path to the data directory')
    parser.add_argument('--work_dir', type=str, default='',
                        help='Experiment directory in which a trained model '
                             'is stored')
    parser.add_argument('--corpus_file', type=str,
                        default='data/pubtator-corpus.pickle',
                        help='Path to training corpus file')
    parser.add_argument('--wbmet_file', type=str, default='',
                        help='Pretrained wbmet embeddings in .vec file')

    # Model design
    parser.add_argument('--n_layers', type=int, default=12,
                        help='Number of total transformer layers')
    parser.add_argument('--n_heads', type=int, default=10,
                        help='Number of heads in multi-head attentions')
    parser.add_argument('--d_head', type=int, default=50,
                        help='Head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='Embedding dimension')
    parser.add_argument('--d_model', type=int, default=500,
                        help='Transformer dimension')
    parser.add_argument('--d_inner', type=int, default=1024,
                        help='Dimension of the inner FF layer')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='Attention probability dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--tgt_len', type=int, default=64,
                        help='Number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=64,
                        help='Number of tokens to predict for evaluation')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='Length of the extended context')
    parser.add_argument('--mem_len', type=int, default=256,
                        help='Length of the retained previous heads')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help=('Apply LayerNorm to the input instead of the '
                              'output'))
    parser.add_argument('--same_length', action='store_true',
                        help='Use the same attn length for all tokens')

    # Optimization
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='Optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='Initial learning rate')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='Momentum value used for SGD')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'inv_sqrt', 'plateau', 'clr'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='Upper limit for learning rate annelaing warmup')
    parser.add_argument('--decay_rate', type=float, default=0.6,
                        help='Decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs (or validations) with no '
                             'improvement after which learning rate will be '
                             'reduced')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='Gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='Only clip the gradient of non-embedding params')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive softmax')
    parser.add_argument('--cutoffs', type=str, default='20000,80000,160000',
                        help='Adaptive cluster cutoffs in comma-separated str')
    parser.add_argument('--div_val', type=int, default=2,
                        help='Divident value for adaptive input and softmax')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='Upper training step limit')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                        help='Min learning rate for cosine scheduler')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='Use the same pos embedding after clamp_len')
    parser.add_argument('--init', default='normal', type=str,
                        choices=['normal', 'uniform'],
                        help='Parameter initializer to use')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='Parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='Parameters initilized by N(0, proj_init_std)')
    parser.add_argument('--init_range', type=float, default=0.1,
                        help='Parameters initialized by U(+-init_range)')

    # Runtime Environment
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--model_name', type=str, default='',
                        help='Unique model identifier used in filenames')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Train on CPU, even if GPUs are available.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Run on a specific GPU')
    parser.add_argument('--random_seed', type=int, default=12345,
                        help='Random seed for all numpy/torch/cuda operations')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='Report interval')
    parser.add_argument('--eval_interval', type=int, default=2000,
                        help='Evaluation interval')
    parser.add_argument('--max_eval_steps', type=int, default=0,
                        help='Limit the evaluation iterations up to max value')


def set_defaults():
    """Set default parameters on runtime"""
    # Unique run-time id
    if not args.model_name:
        args.model_name = time.strftime('%m%d-%H%M')

    # Default Paths
    args.root_dir = Path(__file__).resolve().parent
    if args.data_dir == '':
        args.data_dir = args.root_dir / 'data'

    if args.work_dir == '':
        args.work_dir = args.data_dir / 'trained-models' / 'lmbmet-pubtator'\
                        / args.model_name
    if not args.debug:
        if not args.work_dir.exists():
            args.work_dir.mkdir(parents=True, exist_ok=True)
    args.corpus_file = args.root_dir / args.corpus_file
    if args.wbmet_file != '':
        args.wbmet_file = args.root_dir / args.wbmet_file

    # Random seed
    if args.random_seed >= 0:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # Cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        if torch.cuda.device_count() == 1:
            args.gpu = -1
        torch.cuda.set_device(args.gpu)
        if args.random_seed >= 0:
            torch.cuda.manual_seed(args.random_seed)

    if args.d_embed < 0:
        args.d_embed = args.d_model


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)
    elif args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('LmBMET') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'emb_proj'):
            init_weight(m.emb_proj)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('Conv1d') != -1:
        init_weight(m.weight)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)


###############################################################################
# Training Code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    valid_loss = AverageMeter()
    mems = tuple()

    # Evaluation
    with torch.no_grad():
        start = max(0, len(eval_iter) - args.max_eval_steps)
        start_rnd = np.random.randint(start)
        for batch_num, (data, target) in enumerate(eval_iter):
            if batch_num < start_rnd:
                continue
            if args.cuda:
                data = data.to(args.device)
                target = target.to(args.device)
            loss_mems, hids, pred_hid = model(data, target, *mems)
            loss, mems = loss_mems[0], loss_mems[1:]
            valid_loss.update(loss.mean().item())
            if 0 < args.max_eval_steps <= batch_num:
                pred = model.crit.predict(pred_hid)
                logger.info('Prediction: {}'
                            ''.format(corpus.vocab.decode_symbols(pred[0])))
                logger.info('Target: {}'
                            ''.format(corpus.vocab.decode_symbols(target[0])))
                break

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)

    return valid_loss.avg


def train():
    model.train()
    train_loss = AverageMeter()
    mems = tuple()
    for batch, (data, target) in enumerate(tr_iter):
        if args.cuda:
            data = data.to(args.device)
            target = target.to(args.device)
        model.zero_grad()
        loss_mems, _, _ = model(data, target, *mems)
        loss, mems = loss_mems[0], loss_mems[1:]

        loss = loss.float().mean().type_as(loss)
        loss.backward()
        if args.clip_nonemb:
            torch.nn.utils.clip_grad_norm_(model.layers.parameters(), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # Optimizer step
        optimizer.step()
        train_loss.update(loss.float().item())
        stat['train_step'] += 1

        # Learning rate annealing
        if args.scheduler in ['cosine', 'inv_sqrt', 'clr']:
            scheduler.step(stat['train_step'])

        # Print out train results
        if stat['train_step'] % args.log_interval == 0:
            stat['train_loss'].append((stat['train_step'], train_loss.avg))
            elapsed = time.time() - stat['log_start_time']
            log_str = ('| epoch {} step {} | {}/{} batches | lr {:.3g} '
                       '| ms/batch {:5.2f} | loss {:5.2f} | ppl {:9.3f}'
                       ''.format(stat['epoch'], stat['train_step'],
                                 batch+1, len(tr_iter),
                                 optimizer.param_groups[0]['lr'],
                                 elapsed * 1000 / args.log_interval,
                                 train_loss.avg, math.exp(train_loss.avg)))
            logging.info(log_str)
            train_loss.reset()
            stat['log_start_time'] = time.time()

        # Evaluate and prints out validation results
        if stat['train_step'] % args.eval_interval == 0:
            # Run model evaluation
            stat['eval_start_time'] = time.time()
            val_loss = evaluate(va_iter)
            stat['valid_loss'].append((stat['train_step'], val_loss))
            logging.info('-' * 100)
            log_str = ('| Eval {:3d} at step {:>8d} | time: {:5.2f}s '
                       '| valid loss {:5.2f} | ppl {:9.5f}'
                       ''.format(stat['train_step'] // args.eval_interval,
                                 stat['train_step'],
                                 (time.time() - stat['eval_start_time']),
                                 val_loss, math.exp(val_loss)))
            logging.info(log_str)
            # dev-performance based learning rate annealing
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)

            # Save current loss graph
            save_loss_graph()

            # Run instrinsic embeddings evaluations
            logging.info('-' * 100)
            emb_score = emb_evaluator.eval()
            # Save when emb_score maximized
            if stat['best_emb_score'] < emb_score and batch > 5000:
                stat['best_emb_score'] = emb_score
                logger.info('Saving the best model...')
                with open(args.work_dir/'model.pt', 'wb') as f:
                    torch.save(model, f)

            # # Save the model if the validation loss is the current best
            # if val_loss < stat['best_val_loss'] and batch > 3000:
            #     if not args.debug and batch != 0:
            #         logger.info('Saving the best model...')
            #         with open(args.work_dir/'model.pt', 'wb') as f:
            #             torch.save(model, f)
            #         with open(args.work_dir/'optimizer.pt', 'wb') as f:
            #             torch.save(optimizer.state_dict(), f)
            #     stat['best_val_loss'] = val_loss

        # End trigger
        if stat['train_step'] > args.max_step:
            logging.info('-' * 100)
            logging.info('End of training')
            return True


def save_loss_graph():
    if 'train_loss' not in stat or 'valid_loss' not in stat:
        return
    if not args.debug:
        if not args.work_dir.exists():
            args.work_dir.mkdir(parents=True, exist_ok=True)
        plt.plot(*zip(*stat['train_loss']))
        plt.plot(*zip(*stat['valid_loss']))
        plt.savefig(args.work_dir / 'loss.png')


###############################################################################
# Main script
###############################################################################

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(
        '19-BMET-Embeddings LmBMET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments()
    args = parser.parse_args()
    set_defaults()  # set default parameter values

    # Logger
    logger = logging.getLogger()
    fmt = logging.Formatter('%(asctime)s: [%(message)s]',
                            args.model_name + ' %I:%M%p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(args.work_dir /
                                 'train-{}.log'.format(args.model_name))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    # CUDA
    if args.cuda:
        logger.info('CUDA enabled (GPU {})'.format(args.gpu))

    # Adaptive softmax / embedding  (Baevski et al.)
    cutoffs = list(map(int, args.cutoffs.split(','))) if args.adaptive else None

    # Load Data
    # --------------------------------------------------------------------------
    corpus, (tr_iter, va_iter, ts_iter) = prepare_iterators(args)
    ent_slices, ent_hsz = [], []
    if corpus.vocab.sizes[1] > 0:
        # Assuming that we have a single entity set, and the entities come right
        # after the special codes in the vocabulary
        ent_slices.append(
            slice(corpus.vocab.sizes[0], sum(corpus.vocab.sizes[:2]))
        )
        ent_hsz = [200]  # Hard-coded dimension for entity embeddings
        logger.debug('Resetting ent_slices {}'.format(ent_slices[0]))

    # Build the model
    # --------------------------------------------------------------------------
    model = LmBMET(args, vocab_size=len(corpus.vocab), cutoffs=cutoffs,
                   ent_slices=ent_slices, ent_hsz=ent_hsz)
    model.apply(weights_init)
    model.word_emb.apply(weights_init)
    if args.wbmet_file != '':
        model.read_emb_from_pretrained(corpus.vocab)
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

    # Print out the configuration
    logging.info('=== Options ' + '=' * 88)
    for k, v in args.__dict__.items():
        logging.info('    - {} : {}'.format(k, v))
    logging.info('=' * 100)

    model = model.to(args.device)

    # Optimizer
    # --------------------------------------------------------------------------
    if args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd' or True:  # default
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)

    # Scheduler
    # -- How to adjust the learning rate based on the number of epochs
    # --------------------------------------------------------------------------
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_step, eta_min=args.eta_min
        )
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            """return a multiplier instead of a learning rate"""
            if step == 0 and args.warmup_step == 0:
                return 1.0
            if step > args.warmup_step:
                return 1. / (step ** 0.5)
            return step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.decay_rate, patience=args.patience,
            min_lr=5e-6
        )
    elif args.scheduler == 'clr':
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-7, max_lr=1e-4, step_size_up=3000,
            mode='triangular2', cycle_momentum=False
        )

    # Embeddings Evaluator
    emb_evaluator = EmbeddingEvaluator(
        model=model, vocab=corpus.vocab, mesh_def=corpus.mesh_def,
        eval_sets=['UMNSRS-sim', 'UMNSRS-rel', 'MayoSRS',
                   'MiniMayoSRS-p', 'MiniMayoSRS-c', 'Pedersen-p', 'Pedersen-c',
                   'Hliaoutakis']
    )
    if args.cuda:
        emb_evaluator.model_device = args.device

    # Training loop
    # --------------------------------------------------------------------------
    stat = {
        'epoch': 0,
        'train_step': 0,
        'best_val_loss': 999,
        'best_emb_score': 0,
        'eval_start_time': time.time(),
        'log_start_time': time.time(),
        'train_loss': [],
        'valid_loss': []
    }
    logger.info('Start training...')
    try:
        for epoch in itertools.count(start=1):
            stat['epoch'] = epoch
            end_sig = train()
            if end_sig:
                break
            evaluate(ts_iter)
    except KeyboardInterrupt:
        logging.info('-' * 100)
        logging.info('Exiting from training early')
        save = input('Do you want to overwrite the saved model with the current '
                     'model (y/N)? ')
        if save.lower() == 'y':
            logger.info('Saving the best model...')
            with open(args.work_dir / 'model.pt', 'wb') as f:
                torch.save(model, f)
