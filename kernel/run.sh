#!/bin/bash

rm -r ims

mkdir ims
python3 fine_tune_good_unet.py --sample_size 4 --accumulative_batch_size 4\
 --num_epochs 60 --num_workers 8 --batch_step_one 20\
 --batch_step_two 30 --lr 1e-3\
 --train_dir "The path to your train data"\
 --test_dir "The path to your test data"\
 --model_path "The path to pre-trained sam model"
