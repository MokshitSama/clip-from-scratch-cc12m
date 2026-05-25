"""WebDataset streaming loader for cc12m.

Shard 0020 is held out for the fixed validation set (see build_val_set.py),
so training reads everything except that one shard.
"""
import webdataset as wds
from braceexpand import braceexpand
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from io import BytesIO


SHARDS = (
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0000..0019}.tar")) +
    list(braceexpand("/mnt/md0/cc12m/cc12m-train-{0021..2175}.tar"))
)

IMAGE_SIZE   = 224
MAX_TEXT_LEN = 64
BATCH_SIZE   = 256
NUM_WORKERS  = 8
BUFFER_SIZE  = 4000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
    dataset = (
        wds.WebDataset(
            SHARDS,
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
