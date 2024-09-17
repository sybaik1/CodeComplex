#!/bin/bash

#echo "eval ${2} on k cases"
#CUDA_VISIBLE_DEVICES=$1 python3 eval_k_fold.py --model $2 --folder_name $3.jsonl --model_folder experiments_model/$4
echo "eval ${2} on case random"
CUDA_VISIBLE_DEVICES=$1 python3 eval_model.py --model $2 --valid_path test_random_$3.jsonl --model_folder experiments_model/$4
echo "Eval finished for $2\n"
