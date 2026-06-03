"""Quick throughput benchmark for the two loader strategies.

Times: pure data-loading samples/sec on rank 0 only (no model, no DDP).
We compare wds.WebLoader vs torch.utils.data.DataLoader on the SAME wds
pipeline so the only difference is the loader wrapper.
"""
import os
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import webdataset as wds
from torch.utils.data import DataLoader

import dataset


BATCH = 256
NUM_WORKERS = 8
WARMUP = 5
N_BATCHES = 50


def bench(loader, label, batch_size):
    it = iter(loader)
    # Warmup — first batches eat worker startup cost.
    for _ in range(WARMUP):
        next(it)
    t0 = time.time()
    samples = 0
    for _ in range(N_BATCHES):
        batch = next(it)
        # batch is either a tuple of tensors (DataLoader collated)
        # or a tuple of stacked tensors (wds.batched).
        samples += batch[0].shape[0]
    dt = time.time() - t0
    print(f"{label:>22s}  {samples:>6d} samples in {dt:5.1f}s  =  {samples/dt:6.0f} samples/s")


def build_wds_pipeline(batched_in_pipeline: bool):
    """Same upstream pipeline; only difference is whether we .batched() inside
    (for wds.WebLoader) or not (for torch DataLoader which does its own
    collate)."""
    p = (
        wds.WebDataset(
            dataset.TRAIN_SHARDS,
            shardshuffle=dataset.SHARD_SHUFFLE,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .shuffle(dataset.SHUFFLE_BUFFER)
        .to_tuple("jpg", "txt")
        .map(dataset._preprocess_train, handler=wds.warn_and_continue)
    )
    if batched_in_pipeline:
        p = p.batched(BATCH, partial=False)
    return p


if __name__ == "__main__":
    print(f"BATCH = {BATCH}, NUM_WORKERS = {NUM_WORKERS}, "
          f"WARMUP = {WARMUP}, BENCH = {N_BATCHES} batches\n")

    # --- A: wds.WebLoader (batching in pipeline) ---
    loader_a = wds.WebLoader(
        build_wds_pipeline(batched_in_pipeline=True),
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    bench(loader_a, "wds.WebLoader", BATCH)

    # Recreate to clear workers between benches.
    del loader_a

    # --- B: torch DataLoader (batching via collate) ---
    loader_b = DataLoader(
        build_wds_pipeline(batched_in_pipeline=False),
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    bench(loader_b, "torch DataLoader", BATCH)
