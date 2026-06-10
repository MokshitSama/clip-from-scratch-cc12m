"""Data loading for the precomputed-text pipeline.

Training is a relay: disk -> CPU workers -> GPU. The GPU is the fastest part,
so the pipeline is shaped so it never waits:

  - CPU workers only read and decode. Augmentation and normalization happen
    on the GPU, where they cost ~12ms per 512 images (gpu_transforms.py).
  - Images cross to the GPU as uint8 (1 byte per value), not float32 (4
    bytes). Same picture, a quarter of the PCIe traffic. The float conversion
    happens after the transfer, on the GPU.
  - Captions cost nothing here. Every caption was embedded once, offline, by
    a frozen text model (scripts/embedding_extract.py); a sample's key is its
    row number in that table, so the text side of a sample is one array read.

The embedding table gets copied into /dev/shm at startup. /dev/shm is a
directory backed by RAM: a row read from there takes ~1.5us no matter how
busy the machine is, while a row that has to come from the storage pool can
take ~3.3ms when the file cache is under pressure (measured). 3.3ms is huge
next to a ~380ms training step, and one stalled read stalls a worker, then
that worker's GPU, then — through the loss all_gather — every GPU. The copy
is ~2 minutes, once per boot, and all ranks and workers share the one copy.
"""
import io
import os
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import webdataset as wds
import albumentations as A
from braceexpand import braceexpand
from PIL import Image


# Shards 0020-0219 are held out for validation; training reads everything else.
TRAIN_SHARDS = (
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0000..0019}.tar")) +
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0220..2175}.tar"))
)
VAL_SHARDS = list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0020..0219}.tar"))

EMB_DIR  = os.environ.get("EMB_DIR", "/mnt/md0/cc12m_qwen3emb")
SHM_ROOT = "/dev/shm"

IMAGE_SIZE     = 224
TRAIN_CANVAS   = 256   # workers ship 256x256 uint8; the GPU random-crops to 224
BATCH_SIZE     = 512

# The loader takes batches from its workers in fixed rotation, so one slow
# worker stalls the whole rank while the others sleep on full buffers — they
# are not allowed to deliver out of turn. More workers means each one gets
# more time to finish its batch before its turn comes around again. The box
# has 255 cores; 32 workers per rank leaves plenty idle.
NUM_WORKERS    = 32

# Per worker. Larger buffers shuffle a little better, but the decoded samples
# sitting in them are plain RAM, and past a point they crowd out the OS file
# cache that the tar reads and the embedding table depend on — which made the
# whole pipeline slower, not faster (measured at 4000: ~300GB of buffers,
# cache starved, throughput dropped).
SHUFFLE_BUFFER = 1500

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# CPU-side transforms: fixed-size crop and nothing else. A batch must be one
# tensor, so every image needs the same shape before batching; the random
# augmentation that actually varies per sample runs on the GPU afterwards.
train_tf = A.Compose([
    A.SmallestMaxSize(max_size=TRAIN_CANVAS),
    A.CenterCrop(height=TRAIN_CANVAS, width=TRAIN_CANVAS),
])
val_tf = A.Compose([
    A.SmallestMaxSize(max_size=IMAGE_SIZE),
    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
])


# Opened lazily, once per worker process. A memmap is not read up front: the
# OS pages rows in on first touch and keeps the hot ones cached.
_MM = None
_META = None


def _load_meta(emb_dir=EMB_DIR):
    with open(os.path.join(emb_dir, "meta.json")) as f:
        return json.load(f)


def text_emb_dim(emb_dir=EMB_DIR):
    return _load_meta(emb_dir)["dim"]


def _get_mm(emb_dir):
    global _MM, _META
    if _MM is None:
        _META = _load_meta(emb_dir)
        _MM = np.memmap(os.path.join(emb_dir, _META["file"]), dtype=np.float16,
                        mode="r", shape=(_META["max_rows"], _META["dim"]))
    return _MM


def stage_embeddings_to_shm(src=EMB_DIR, shm_root=SHM_ROOT):
    """Copy the embedding table into /dev/shm (RAM) once. Idempotent.

    Rows are read in random order, thousands per second. From RAM that read
    is ~1.5us and can never be evicted; from the pool it can be ~3.3ms when
    the file cache is squeezed, and those misses froze random workers for
    seconds at a time.

    Rank 0 calls this and does the copy; the other ranks call it after a
    barrier and hit the already-staged path. If /dev/shm lacks the space, we
    warn and fall back to reading from the pool — slower, but training runs.
    """
    import shutil

    src = str(src)
    dst = os.path.join(shm_root, os.path.basename(src.rstrip("/")))
    meta = _load_meta(src)
    mm_src = os.path.join(src, meta["file"])
    mm_dst = os.path.join(dst, meta["file"])
    need = os.path.getsize(mm_src)

    if os.path.exists(mm_dst) and os.path.getsize(mm_dst) == need:
        return dst

    free = shutil.disk_usage(shm_root).free
    if free < need * 1.05:
        print(f"[stage_emb] WARNING: {shm_root} has {free/1e9:.0f}GB free, "
              f"need {need/1e9:.0f}GB — staying on {src} (slow cold reads!)",
              flush=True)
        return src

    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, "meta.json"), os.path.join(dst, "meta.json"))
    tmp = mm_dst + ".tmp"
    shutil.copyfile(mm_src, tmp)
    os.rename(tmp, mm_dst)        # atomic: a half-finished copy is never visible under the real name
    return dst


def _make_preprocess(emb_dir, tf, draft_size):
    def preprocess(sample):
        key, jpg = sample
        mm = _get_mm(emb_dir)
        # The key is the row number — no lookup table needed. Stays float16:
        # half the transfer bytes, and the GPU computes in bf16 anyway.
        emb = np.array(mm[int(key)])
        if not emb.any():        # all-zero row = caption was empty at extraction; drop the pair
            return None
        img = Image.open(io.BytesIO(jpg))
        # JPEG decoders can produce a 1/2, 1/4 or 1/8 scale image directly
        # while decoding, nearly free. We downsize to ~256 right after anyway,
        # so decoding a 4000px photo at full resolution is wasted work. This
        # roughly halved decode time, and evened it out across images — big
        # photos no longer make one worker's batch take much longer than
        # everyone else's.
        img.draft("RGB", (draft_size, draft_size))
        img = np.array(img.convert("RGB"))
        img = tf(image=img)["image"]
        return torch.from_numpy(img), torch.from_numpy(emb)
    return preprocess


def _passthrough_node(src, group=None):
    yield from src


def build_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                 shuffle_buffer=SHUFFLE_BUFFER, emb_dir=EMB_DIR):
    pipeline = (
        wds.WebDataset(
            TRAIN_SHARDS,
            shardshuffle=100,
            nodesplitter=wds.split_by_node,        # each GPU reads its own slice of the shards
            workersplitter=wds.split_by_worker,    # and each worker its own slice of that
            handler=wds.warn_and_continue,         # a corrupt sample logs a warning instead of killing the run
            empty_check=False,
        )
        .shuffle(shuffle_buffer)
        .to_tuple("__key__", "jpg")
        .map(_make_preprocess(emb_dir, train_tf, draft_size=TRAIN_CANVAS),
             handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=False)        # the contrastive loss needs every batch the same size
    )
    # pin_memory=False is deliberate. The built-in pinning thread page-locks
    # fresh memory for every batch, all run long, and that crashed here with
    # "CUDA error: invalid argument" (the repo hit the same class of failure
    # before — commit e5badff). Instead we pin a few buffers once at startup
    # and reuse them: gpu_transforms.PinnedPrefetcher.
    return wds.WebLoader(pipeline, batch_size=None, num_workers=num_workers,
                         pin_memory=False, persistent_workers=num_workers > 0,
                         prefetch_factor=2 if num_workers > 0 else None)


def build_val_loader(batch_size=512, num_workers=16, emb_dir=EMB_DIR):
    """One deterministic pass: no shuffle, no augmentation, same order every
    time, so eval numbers are comparable between checkpoints."""
    pipeline = (
        wds.WebDataset(
            VAL_SHARDS,
            shardshuffle=False,
            resampled=False,
            nodesplitter=_passthrough_node,        # eval runs on rank 0 only; it reads ALL val shards
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .to_tuple("__key__", "jpg")
        .map(_make_preprocess(emb_dir, val_tf, draft_size=IMAGE_SIZE),
             handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=True)
    )
    return wds.WebLoader(pipeline, batch_size=None, num_workers=num_workers,
                         pin_memory=False, persistent_workers=False,
                         prefetch_factor=2 if num_workers > 0 else None)


if __name__ == "__main__":
    import time
    print(f"EMB_DIR={EMB_DIR}  dim={text_emb_dim()}")
    loader = build_loader(batch_size=64, num_workers=2)
    t0 = time.time()
    for i, (imgs, emb) in enumerate(loader):
        print(f"batch {i}: imgs {tuple(imgs.shape)} {imgs.dtype} | "
              f"text_emb {tuple(emb.shape)} {emb.dtype} | "
              f"emb norm {emb.float().norm(dim=1).mean():.3f}  ({time.time()-t0:.2f}s)")
        if i >= 2:
            break
        t0 = time.time()
