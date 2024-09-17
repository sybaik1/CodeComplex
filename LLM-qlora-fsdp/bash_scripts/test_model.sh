#!/bin/bash
echo "python3 ./scripts/test_model.py --model $1 ${@:2}"
python3 ./scripts/test_model.py --model $1 ${@:2}
