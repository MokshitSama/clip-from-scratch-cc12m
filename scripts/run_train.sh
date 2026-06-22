#!/bin/bash
# Launch train_precomp.py on the 6 5090s under accelerate.
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID is REQUIRED: without it CUDA's default
# FASTEST_FIRST ordering would reinterpret CUDA_VISIBLE_DEVICES, which
# silently swaps the 3090 (GPU 4 in nvidia-smi) into one of the visible
# slots and causes an illegal-memory-access crash mid-training.
#
# Usage:
#     VERSION=7 ./scripts/run_train.sh
#     VERSION=7 ./scripts/run_train.sh --config alt.yaml   # extra args go through to train_precomp.py
#
# Requires VERSION env var (so we never accidentally overwrite an old runs/v{N}/).

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VERSION:-}" ]]; then
    echo "ERROR: VERSION env var is required (e.g. VERSION=7 ./scripts/run_train.sh)" >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6   # six 5090s; GPU 4 (3090) skipped

echo "Launching train_precomp.py:"
echo "  CUDA_DEVICE_ORDER   = $CUDA_DEVICE_ORDER"
echo "  CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
echo "  VERSION             = $VERSION"
echo "  output -> runs/v$VERSION/"
echo

accelerate launch \
    --num_processes 6 \
    --gpu_ids all \
    train_precomp.py "$@"
