#!/bin/bash

langs="
javadead
pythondead
"

i=0;
for i in {0..3}; do
  for lang in $langs; do
    python3 dead_remove_${lang}.py --file_name test_${i}_fold_${lang}_data.jsonl
  done
done
