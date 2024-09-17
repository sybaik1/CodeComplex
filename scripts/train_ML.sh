#!/bin/bash

lang=${2}

datas="
${lang}_data
"

models="
${lang}_data
"

echo "eval ${2}\n\n"
for data1 in $datas; do
  for data2 in $models; do
    echo "eval on $data1 with $data2"
    python3 MLbase_k.py --model ${1} --train_suffix ${data2} --test_suffix ${data1} --lang ${2} > result/${1}_k_${data2}_${data1}.txt
    #python3 MLbase_k.py --model RandomForest --train_suffix $data2 --test_suffix $data1 --lang ${2} > result/RandomForest_k_$1_$2.txt
    #python3 MLbase_k.py --model DecisionTree --train_suffix $data2 --test_suffix $data1 --lang ${2} > result/DecisionTree_k_$1_$2.txt
  done
done
