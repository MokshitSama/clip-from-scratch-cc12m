"""Build a fixed 5000-sample validation set from cc12m shard 0020.

Saving the preprocessed tensors to disk decouples evaluation speed from data-
loading speed and makes eval-to-eval comparisons reproducible.
"""
import io
import sys
import time
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


SHARD_PATH = "/mnt/md0/cc12m/cc12m-train-0020.tar"
OUT_PATH   = Path(__file__).parent / "val_set_cc12m_0020_5000.pt"
N_SAMPLES  = 5000
IMAGE_SIZE = 224
MAX_TEXT_LEN = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main():
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists. Delete it to rebuild.")
        return

    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # cc12m shards group .jpg + .txt by common stem (000000000.jpg / 000000000.txt).
    # Walk the tar, pair them up by stem, take the first N valid pairs.
    pending = {}
    images, input_ids, attn_masks, captions = [], [], [], []

    t0 = time.time()
    with tarfile.open(SHARD_PATH, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem, _, ext = member.name.rpartition(".")
            if ext not in ("jpg", "txt"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            pending.setdefault(stem, {})[ext] = f.read()

            entry = pending[stem]
            if "jpg" in entry and "txt" in entry:
                pair = pending.pop(stem)
                try:
                    img = Image.open(io.BytesIO(pair["jpg"])).convert("RGB")
                    img_t = image_transform(img)

                    text = pair["txt"].decode("utf-8").strip()
                    if not text:
                        continue
                    tok = tokenizer(text, max_length=MAX_TEXT_LEN, padding="max_length",
                                    truncation=True, return_tensors="pt")
                except Exception:
                    continue   # skip corrupt entries — same policy as train loader

                images.append(img_t)
                input_ids.append(tok["input_ids"].squeeze(0))
                attn_masks.append(tok["attention_mask"].squeeze(0))
                captions.append(text)

                if len(images) >= N_SAMPLES:
                    break

    if len(images) < N_SAMPLES:
        sys.exit(f"only collected {len(images)} / {N_SAMPLES} samples — shard may be too corrupted")

    blob = {
        "images":         torch.stack(images),
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attn_masks),
        "captions":       captions,
        "source_shard":   SHARD_PATH,
        "n_samples":      N_SAMPLES,
    }
    torch.save(blob, OUT_PATH)

    print(f"collected {N_SAMPLES} pairs in {time.time() - t0:.1f}s")
    print(f"saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
