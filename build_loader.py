"""WebDataset streaming loaders for cc12m.

Shards 0020-0219 (200 shards, ~1M pairs) are held out as the validation set.
Training reads the other 1976 shards (~9.96M pairs).
"""
import webdataset as wds
from braceexpand import braceexpand
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from io import BytesIO


# Train: everything except the 200-shard val range.
TRAIN_SHARDS = (
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0000..0019}.tar")) +
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0220..2175}.tar"))
)
# Val: ~1M held-out pairs. Streamed at eval time (too big to preload).
VAL_SHARDS = list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0020..0219}.tar"))

# Back-compat alias for callers that imported the old name.
SHARDS = TRAIN_SHARDS

IMAGE_SIZE   = 224
MAX_TEXT_LEN = 64
BATCH_SIZE   = 256
NUM_WORKERS  = 8
BUFFER_SIZE  = 4000

# Tokenizer must match the text encoder used in model.py. BGE-base uses the
# BertTokenizer with the same vocab as bert-base-uncased, so the tokenization
# is byte-identical to the previous setup — but we point at the BGE repo so
# any future divergence stays in sync.
TEXT_ENCODER = "BAAI/bge-base-en-v1.5"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)


def preprocess(sample):
    img_bytes, text_bytes = sample
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        image_tensor = image_transform(img)

        text = text_bytes.decode("utf-8").strip()
        if not text:
            return None

        tokens = tokenizer(
            text,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return image_tensor, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)
    except Exception:
        return None


def build_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Training loader. Infinite resampled stream, sharded across DDP ranks."""
    dataset = (
        wds.WebDataset(
            TRAIN_SHARDS,
            shardshuffle=False,
            resampled=True,
            nodesplitter=wds.split_by_node,   # each DDP rank gets a disjoint shard slice
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .shuffle(BUFFER_SIZE)
        .to_tuple("jpg", "txt")
        .map(preprocess)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=False)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=False)


def _no_node_split(src, group=None):
    """Pass-through nodesplitter for the val loader.

    Webdataset's default (`single_node_only`) raises if world_size > 1, even
    though we only build the val loader on rank 0. We want rank 0 to see ALL
    val shards, not its 1/world_size slice, so the splitter just yields
    everything through.
    """
    yield from src


def build_val_loader(batch_size=512, num_workers=4):
    """Validation loader. Finite, deterministic, single pass through VAL_SHARDS.

    No shuffle, no resampling — eval runs on rank 0 only and expects the
    exact same sequence of pairs every time so metrics are reproducible.
    """
    dataset = (
        wds.WebDataset(
            VAL_SHARDS,
            shardshuffle=False,
            resampled=False,
            nodesplitter=_no_node_split,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .to_tuple("jpg", "txt")
        .map(preprocess)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=True)
    )
    return wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=False)


if __name__ == "__main__":
    import time
    loader = build_loader()
    t0 = time.time()
    for i, (imgs, ids, mask) in enumerate(loader):
        dt = time.time() - t0
        print(f"batch {i} ({dt:.2f}s)  imgs {tuple(imgs.shape)}  caption[0]: {tokenizer.decode(ids[0], skip_special_tokens=True)[:80]!r}")
        if i >= 2:
            break
        t0 = time.time()
