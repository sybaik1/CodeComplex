#!/bin/bash
script_name=$(basename $1)
while read -r model_name; do
    echo ${model_name}
    /bin/bash $1 ${model_name} ${@:2}
done<modellist.txt
