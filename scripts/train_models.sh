#!/bin/bash

i=0;
for i in {0..3}
do
	echo "training ${2} on case ${i}"
	CUDA_VISIBLE_DEVICES=$1 python3 train.py --model $2 --train_path train_${i}_fold_$3.jsonl --test_path test_${i}_fold_$3.jsonl --model_folder models/$3 ${4:+--epoch $4}
done
echo "training ${2} on case random"
CUDA_VISIBLE_DEVICES=$1 python3 train.py --model $2 --train_path train_random_$3.jsonl --test_path test_random_$3.jsonl --model_folder models/$3 ${4:+--epoch $4}

echo "Training finished for $2\n"
