#!/bin/bash
echo "python3 ./scripts/scoring.py --model $1 ${@:2}"
python3 ./scripts/scoring.py --model $1 ${@:2}
