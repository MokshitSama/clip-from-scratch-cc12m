#!/bin/bash
# Launch 6 parallel embedding extractors, one per 5090.
# Use PCI_BUS_ID so CUDA_VISIBLE_DEVICES matches nvidia-smi indexing.
# Skip GPU 4 (the 3090) — slower hardware would gate the whole job.

export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPUS=(0 1 2 3 5 6)
WORLD_SIZE=${#GPUS[@]}

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    GPU=${GPUS[$RANK]}
    echo "Launching rank $RANK on GPU $GPU"
    RANK=$RANK \
    WORLD_SIZE=$WORLD_SIZE \
    CUDA_VISIBLE_DEVICES=$GPU \
    python scripts/embedding_extract.py \
        > /tmp/extract_rank_${RANK}.log 2>&1 &
done

echo "Launched ${WORLD_SIZE} processes. PIDs: $(jobs -p)"
echo "Monitor: tail -f /tmp/extract_rank_*.log"
echo "Waiting for all to finish..."
wait
echo "All ranks done."
