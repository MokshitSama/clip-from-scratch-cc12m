"""cc12m streaming pipeline: WebDataset end-to-end (parsing + loader).

WebDataset spawns N worker processes via the same DataLoader-style machinery
as torch (wds.WebLoader is literally a thin subclass of
torch.utils.data.DataLoader). So data loading is concurrent across CPU
cores — each worker runs its own tar-iterate, decode, augment, tokenize
pipeline in parallel, and yields batches to the main process.

DDP shard splitting:
    nodesplitter=split_by_node + workersplitter=split_by_worker (with
    resampled=False, the default) gives **strictly disjoint** shard slices
    across (rank x worker) — every shard is consumed by exactly one worker
    process per epoch.

Augmentation:
    Train uses albumentations with RandomResizedCrop / HFlip / ColorJitter
    — standard for image-text contrastive (cc12m-scale wants more aug than
    the original CLIP recipe of "random square crop only" because cc12m is
    ~100x smaller than WIT400M).
    Val uses a deterministic Resize+CenterCrop — must match exactly across
    runs so eval is reproducible.
"""
import io
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import webdataset as wds
import albumentations as A
from albumentations.pytorch import ToTensorV2
from braceexpand import braceexpand
from PIL import Image
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Shard ranges + encoder/tokenizer choice.
# ---------------------------------------------------------------------------
TRAIN_SHARDS = (
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0000..0019}.tar")) +
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0220..2175}.tar"))
)
VAL_SHARDS = list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0020..0219}.tar"))

TEXT_ENCODER   = "BAAI/bge-base-en-v1.5"

IMAGE_SIZE     = 224
MAX_TEXT_LEN   = 64
BATCH_SIZE     = 256
NUM_WORKERS    = 8
SHUFFLE_BUFFER = 4000
SHARD_SHUFFLE  = 100             # window for wds shard-level shuffle

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# Tokenizer — module-level so it gets pickled into worker processes (cheap;
# the tokenizer is just a small Python object).
tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)


# ---------------------------------------------------------------------------
# Transforms.
# ---------------------------------------------------------------------------
train_tf = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0, p=0.5),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# Val: deterministic, no augmentation.
val_tf = A.Compose([
    A.SmallestMaxSize(max_size=IMAGE_SIZE),
    A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def _decode_and_transform(jpg_bytes: bytes, txt_bytes: bytes, transform):
    """Decode JPEG, transform, tokenize. Returns (img, ids, mask) or raises
    (raised exceptions get caught by wds.warn_and_continue upstream)."""
    img = np.array(Image.open(io.BytesIO(jpg_bytes)).convert("RGB"))
    img = transform(image=img)["image"]

    text = txt_bytes.decode("utf-8").strip() or " "       # empty → space (still tokenizes)
    tok = tokenizer(
        text,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return img, tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)


def _preprocess_train(sample):
    jpg, txt = sample
    return _decode_and_transform(jpg, txt, train_tf)


def _preprocess_val(sample):
    jpg, txt = sample
    return _decode_and_transform(jpg, txt, val_tf)


def _passthrough_node(src, group=None):
    """No-op nodesplitter for val (rank 0 only — wants every shard)."""
    yield from src


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
def build_loader(batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 shuffle_buffer: int = SHUFFLE_BUFFER) -> wds.WebLoader:
    """Train loader. Strict per-(rank, worker) disjoint shard splits, shuffled
    shard order each epoch, sample-level shuffle buffer. Batching happens
    inside the pipeline (`.batched(...)`) — wds.WebLoader is told
    `batch_size=None` because batches are already constructed upstream."""
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
        .to_tuple("jpg", "txt")
        .map(_preprocess_train, handler=wds.warn_and_continue)
        .batched(batch_size, partial=False)        # contrastive needs uniform B
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
        .to_tuple("jpg", "txt")
        .map(_preprocess_val, handler=wds.warn_and_continue)
        .batched(batch_size, partial=True)         # keep final partial batch on val
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
    for i, (imgs, ids, mask) in enumerate(loader):
        dt = time.time() - t0
        print(f"  batch {i}: imgs {tuple(imgs.shape)} {imgs.dtype}  "
              f"ids {tuple(ids.shape)}  mask {tuple(mask.shape)}  ({dt:.2f}s)")
        print(f"    img stats: mean={imgs.float().mean():.3f}  std={imgs.float().std():.3f}")
        if i >= 2:
            break
        t0 = time.time()

    val_loader = build_val_loader(batch_size=256, num_workers=2)
    print("\nfirst 2 val batches:")
    t0 = time.time()
    for i, (imgs, ids, mask) in enumerate(val_loader):
        dt = time.time() - t0
        print(f"  batch {i}: imgs {tuple(imgs.shape)}  ({dt:.2f}s)")
        if i >= 1:
            break
        t0 = time.time()
