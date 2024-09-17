#!/bin/bash
echo "python3 ./scripts/confusion_mat.py --model $1 ${@:2}"
python3 ./scripts/confusion_mat.py --model $1 ${@:2}
