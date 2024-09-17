#!/bin/bash
accelerate launch --config_file accelerate_config_fsdp.yaml scripts/accelerate_fsdp_qlora.py --model $1 ${@:2}
