#!/bin/bash
echo 'Running LmBMET training...'
python train_lmbmet.py \
    --corpus_file data/pubtator/pubtator-corpus-1210_1448.pickle \
    --adaptive \
    --div_val 2 \
    --batch_size 16 \
    --n_layer 12 \
    --n_head 6 \
    --d_head 256 \
    --d_inner 1024 \
    --init uniform \
    --dropout 0.2 \
    --dropatt 0.1 \
    --clip 0.1 \
    --optim adam \
    --lr 0.0002 \
    --eta_min 1e-8 \
    --max_step 100000 \
    --max_eval_steps 1000 \
    --tgt_len 128 \
    --eval_tgt_len 196 \
    --mem_len 24 \
    --log_interval 200 \
    --eval_interval 1000 \
    --gpu 1 \
    ${@:1}
