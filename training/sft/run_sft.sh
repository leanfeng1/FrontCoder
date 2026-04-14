#!/bin/bash
# FrontCoder SFT Training Script
#
# This script trains the SFT model for front-end code generation.
# Uses veRL framework for distributed training.
#
# Usage:
#   bash run_sft.sh [--config CONFIG_PATH]
#
# Requirements:
#   - veRL framework (pip install verl)
#   - 8x H800 GPUs recommended

set -x

# Default configuration
nproc_per_node=8
CONFIG_PATH="${CONFIG_PATH:-$(dirname $0)}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --nproc)
            nproc_per_node="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "FrontCoder SFT Training"
echo "========================================"
echo "Config Path: $CONFIG_PATH"
echo "GPUs: $nproc_per_node"
echo "========================================"

# Run training
torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$nproc_per_node \
    --master_port=12345 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=$CONFIG_PATH \
    --config-name=sft_config.yaml
