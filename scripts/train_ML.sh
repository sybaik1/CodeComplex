#!/bin/bash

python3 MLbase_k.py --model SVM --data_suffix $1 > result/SVM_k_$1.txt
python3 MLbase_k.py --model RandomForest --data_suffix $1 > result/RandomForest_k_$1.txt
python3 MLbase_k.py --model DecisionTree --data_suffix $1 > result/DecisionTree_k_$1.txt
