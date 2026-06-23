import io
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import webdataset as wds

from braceexpand import braceexpand
from PIL import Image


# ---------------------------------------------------------------------------
# Shard ranges + encoder/tokenizer choice.
# ---------------------------------------------------------------------------
TRAIN_SHARDS = (
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0000..0019}.tar")) +
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0220..2175}.tar"))
)
VAL_SHARDS = list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0020..0219}.tar"))

CROP_SIZE     = 256
BATCH_SIZE     = 64        # v8 SigLIP: smaller image batch (image fwd is the bottleneck);
                           # contrastive difficulty comes from `n_negs` random text rows
                           # pulled from /dev/shm each step (see train_utils.train_steps).
NUM_WORKERS    = 8
SHUFFLE_BUFFER = 4000
SHARD_SHUFFLE  = 100             # window for wds shard-level shuffle

def _decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    """JPEG → uint8 HWC numpy, short-side = CROP_SIZE, then center-cropped square.
    Uses libjpeg draft mode for fast decode (1/2, 1/4 or 1/8 scale during decode)."""

    img = Image.open(io.BytesIO(jpg_bytes))
    img.draft("RGB", (CROP_SIZE, CROP_SIZE))
    img = img.convert("RGB")

    w, h = img.size
    if w < h:
        new_w, new_h = CROP_SIZE, int(round(h * CROP_SIZE / w))

    else:
        new_w, new_h = int(round(w * CROP_SIZE/h)), CROP_SIZE

    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - CROP_SIZE) // 2
    top = (new_h - CROP_SIZE) // 2
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
                    #left, top,    Right,          bottom

    return np.asarray(img, dtype=np.uint8)


def _preprocess(sample):
    key, jpg = sample
    return int(key), _decode_jpg(jpg)

def _collate(samples):
    """List of (idx, img) → (imgs_batch, idx_batch) as numpy arrays."""
    keys, imgs = zip(*samples)
    return np.stack(imgs, axis=0), np.asarray(keys, dtype=np.int64)

def _passthrough_node(src, group=None):
    """No-op nodesplitter for val (rank 0 only — wants every shard)."""
    yield from src


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
def build_loader(batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 shuffle_buffer: int = SHUFFLE_BUFFER) -> wds.WebLoader:
    """Train loader: disjoint per (rank, worker) shards, sample-level shuffle."""

    pipeline = (
        wds.WebDataset(
            TRAIN_SHARDS,
            shardshuffle=SHARD_SHUFFLE,
            nodesplitter=wds.split_by_node,        # split shards across DDP ranks
            workersplitter=wds.split_by_worker,    # then across loader workers
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .shuffle(shuffle_buffer)
        .to_tuple("__key__", "jpg")
        .map(_preprocess, handler=wds.warn_and_continue)
        .batched(batch_size, collation_fn=_collate, partial=False)        # contrastive needs uniform B
    )
    return wds.WebLoader(
        pipeline,
        batch_size=None,                           # already batched in the pipeline
        num_workers=num_workers,
        pin_memory=False,         # disabled: hit CUDA "invalid format" with wds + pinned tensors
        persistent_workers=num_workers > 0,
    )


def build_val_loader(batch_size: int = 512,
                     num_workers: int = 12) -> wds.WebLoader:
    """Val loader. Single deterministic pass over VAL_SHARDS on rank 0 only."""
    pipeline = (
        wds.WebDataset(
            VAL_SHARDS,
            shardshuffle=False,
            resampled=False,
            nodesplitter=_passthrough_node,        # rank 0 wants ALL val shards
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .to_tuple("__key__", "jpg")
        .map(_preprocess, handler=wds.warn_and_continue)
        .batched(batch_size, collation_fn=_collate, partial=True)         # keep final partial batch on val
    )
    return wds.WebLoader(
        pipeline,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,         # disabled: hit CUDA "invalid format" with wds + pinned tensors
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    print(f"TRAIN_SHARDS: {len(TRAIN_SHARDS)}  VAL_SHARDS: {len(VAL_SHARDS)}")

    loader = build_loader(batch_size=64, num_workers=2)
    print("\nfirst 3 train batches:")
    t0 = time.time()
    for i, (imgs, idx) in enumerate(loader):
        dt = time.time() - t0
        print(f"  batch {i}: imgs {imgs.shape} {imgs.dtype}  "
            f"idx {idx.shape} {idx.dtype}  idx[:3]={idx[:3].tolist()}  ({dt:.2f}s)")
        if i >= 2:
            break
        t0 = time.time()