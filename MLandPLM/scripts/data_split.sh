#!/bin/bash

python3 data_processing/split.py --file_name $1.jsonl --size 0.8 --lang $2
python3 data_processing/split_n_fold.py --file_name $1.jsonl --size 0.8 --fold 4 --lang $2

i=0;
for i in {0..4}
do
	python3 data_processing/data_split_by_complexity.py --file_name test_${i}_fold_$1.jsonl
	python3 data_processing/data_split_by_length.py --file_name test_${i}_fold_$1.jsonl --lang $2
done
