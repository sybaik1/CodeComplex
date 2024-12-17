#!/bin/bash

lang=${3}

datas="
${lang}dead_data
"
models="
${lang}_data
${lang}Aug_data
"

echo "eval ${2}\n\n"
for data1 in $datas; do
  for data2 in $models; do
    echo "eval on $data1 with $data2"
#    CUDA_VISIBLE_DEVICES=$1 python3 eval_k_fold.py --model $2 --folder_name $data1.jsonl --model_folder experiments_model/$data2 
    CUDA_VISIBLE_DEVICES=$1 python3 eval_model.py --model $2 --valid_path test_random_$data1.jsonl --model_folder experiments_model/$data2
  done
done
echo "Eval finished for $2\n\n"
