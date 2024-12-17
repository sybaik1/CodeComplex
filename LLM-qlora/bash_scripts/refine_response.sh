#!/bin/bash
echo "python3 ./scripts/refine_response.py --model $1 ${@:2}"
python3 ./scripts/refine_response.py --model $1 ${@:2}
