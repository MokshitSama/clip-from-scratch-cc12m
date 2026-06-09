"""Benchmark v6 against OpenAI CLIP and OpenCLIP cc12m.

Uses the cached 5000-pair set from cc12m shard 0020. All models score on
the SAME 5000 images / 5000 captions — only the preprocessing differs
(each model's own normalization + tokenizer).

Caveats worth knowing when reading the results:

- Shard 0020 is in our held-out set (we train on shards 0000-0019 +
  0220-2175). For OpenAI CLIP it's never seen. For OpenCLIP cc12m it
  COULD be in training data, since they used all of cc12m. Their
  advantage there is real but bounded — cc12m has 12M pairs, the
  chance of having memorized a specific (image, caption) is low.
- All three reference models are ViT / RN, not EfficientNet B1. Different
  architectures and parameter counts. We're comparing recipes more than
  fair apples-to-apples.
- 5000 candidates is the size used in the CLIP paper's COCO eval; R@1 at
  this scale is the standard metric people report.

Run with:
    CUDA_VISIBLE_DEVICES=4 python benchmark.py
"""
import os

# IMPORTANT: pin to the idle 3090 BEFORE torch imports / CUDA inits, so we
# don't fight v6 training for VRAM on a 5090.
# CUDA_DEVICE_ORDER=PCI_BUS_ID forces CUDA to match nvidia-smi's index
# order (otherwise CUDA picks FASTEST_FIRST and the 3090 ends up at a
# different index than what nvidia-smi shows).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using device: {name}  ({total_gb:.1f} GB)")
    assert total_gb < 30, f"Expected 24 GB 3090, got {total_gb:.1f} GB — wrong GPU. Set CUDA_VISIBLE_DEVICES=4 manually."
import torch.nn.functional as F
import open_clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CLIPModel


DEVICE   = "cuda"
VAL_PATH = Path(__file__).parent / "val_set_cc12m_0020_5000.pt"
CKPT_DIR = Path(__file__).parent / "runs" / "v6" / "checkpoints"

# Our preprocessing used ImageNet stats; CLIP uses its own.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN     = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD      = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def renormalize_imagenet_to_clip(imgs: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization, redo with CLIP normalization."""
    m_i, s_i = IMAGENET_MEAN.to(imgs.device), IMAGENET_STD.to(imgs.device)
    m_c, s_c = CLIP_MEAN.to(imgs.device),     CLIP_STD.to(imgs.device)
    raw = imgs * s_i + m_i                      # back to [0, 1]
    return (raw - m_c) / s_c                    # forward with CLIP stats


def find_latest_ckpt() -> Path:
    """Pick the highest-step checkpoint currently on disk."""
    ckpts = sorted(CKPT_DIR.glob("ckpt_step*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {CKPT_DIR}")
    return ckpts[-1]


@torch.no_grad()
def embed_ours(model, val, bs=64):
    """Embed with our v6 CLIPModel (ImageNet preprocessing, BGE tokens)."""
    images    = val["images"].to(DEVICE)
    input_ids = val["input_ids"].to(DEVICE)
    mask      = val["attention_mask"].to(DEVICE)

    img_out, txt_out = [], []
    for i in range(0, images.shape[0], bs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ie, te, _ = model(images[i:i+bs], input_ids[i:i+bs], mask[i:i+bs])
        img_out.append(ie.float())
        txt_out.append(te.float())
    return torch.cat(img_out), torch.cat(txt_out)


@torch.no_grad()
def embed_openclip(model, tokenizer, val, bs=32, max_text_len=77):
    """Embed with an open_clip model.

    Re-normalize stored ImageNet-normalized tensors to CLIP stats.
    Re-tokenize raw captions with each model's BPE tokenizer.
    """
    imgs = renormalize_imagenet_to_clip(val["images"].to(DEVICE))
    # open_clip tokenizers truncate to context_length automatically.
    text_tokens = tokenizer(val["captions"]).to(DEVICE)
    if text_tokens.shape[1] > max_text_len:
        text_tokens = text_tokens[:, :max_text_len]

    img_out, txt_out = [], []
    for i in range(0, imgs.shape[0], bs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ie = model.encode_image(imgs[i:i+bs])
            te = model.encode_text(text_tokens[i:i+bs])
        img_out.append(F.normalize(ie.float(), dim=-1))
        txt_out.append(F.normalize(te.float(), dim=-1))
    return torch.cat(img_out), torch.cat(txt_out)


def retrieval_metrics(img_emb, txt_emb):
    """Standard R@{1,5,10} + mean_rank for both directions on an NxN sim matrix."""
    n = img_emb.shape[0]
    labels = torch.arange(n, device=img_emb.device)
    sim = img_emb @ txt_emb.T
    out = {}
    for direction, scores in [("i2t", sim), ("t2i", sim.T)]:
        for k in (1, 5, 10):
            topk = scores.topk(k, dim=1).indices
            out[f"r@{k}_{direction}"] = (topk == labels[:, None]).any(dim=1).float().mean().item() * 100
        sorted_idx = scores.argsort(dim=1, descending=True)
        ranks = (sorted_idx == labels[:, None]).float().argmax(dim=1) + 1
        out[f"mean_rank_{direction}"] = ranks.float().mean().item()
    return out


def main():
    print(f"Loading val set: {VAL_PATH.name}")
    val = torch.load(VAL_PATH, weights_only=False)
    n = val["images"].shape[0]
    print(f"  {n} pairs\n")

    results = {}
    sizes   = {}

    # --- Our v6 ---
    ckpt_path = find_latest_ckpt()
    print(f"Loading our v6 ({ckpt_path.name})...")
    our = CLIPModel().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = our.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  step: {ckpt.get('step')}  missing: {len(missing)}  unexpected: {len(unexpected)}")
    our.eval()

    n_params_v6 = sum(p.numel() for p in our.parameters())
    sizes["v6 (ours)"] = n_params_v6

    print("Embedding with v6...")
    t0 = time.time()
    img, txt = embed_ours(our, val)
    print(f"  done in {time.time() - t0:.1f}s")
    results["v6 (ours)"] = retrieval_metrics(img, txt)
    del our, img, txt
    torch.cuda.empty_cache()

    # --- Reference models ---
    refs = [
        ("OpenAI CLIP ViT-B/16", "ViT-B-16",  "openai"),
        ("OpenAI CLIP ViT-L/14", "ViT-L-14",  "openai"),
        ("OpenCLIP RN50 cc12m",  "RN50",      "cc12m"),
    ]
    for label, arch, pretrained in refs:
        print(f"\nLoading {label} ({arch}, pretrained={pretrained})...")
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(arch)
        model = model.to(DEVICE).eval()

        sizes[label] = sum(p.numel() for p in model.parameters())

        print(f"Embedding with {label}...")
        t0 = time.time()
        img, txt = embed_openclip(model, tokenizer, val)
        print(f"  done in {time.time() - t0:.1f}s")
        results[label] = retrieval_metrics(img, txt)
        del model, img, txt
        torch.cuda.empty_cache()

    # --- Print comparison table ---
    print()
    print("=" * 100)
    print(f"Benchmark: {n}-pair held-out set from cc12m shard 0020")
    print("=" * 100)
    header = (f"{'Model':<24s} {'Params':>10s}   "
              f"{'i2t R@1':>8s} {'R@5':>7s} {'R@10':>7s}   "
              f"{'t2i R@1':>8s} {'R@5':>7s} {'R@10':>7s}   "
              f"{'meanR':>7s}")
    print(header)
    print("-" * 100)
    for label, m in results.items():
        mr_avg = (m["mean_rank_i2t"] + m["mean_rank_t2i"]) / 2
        params_m = sizes[label] / 1e6
        print(f"{label:<24s} {params_m:>8.1f}M   "
              f"{m['r@1_i2t']:>7.2f}% {m['r@5_i2t']:>6.2f}% {m['r@10_i2t']:>6.2f}%   "
              f"{m['r@1_t2i']:>7.2f}% {m['r@5_t2i']:>6.2f}% {m['r@10_t2i']:>6.2f}%   "
              f"{mr_avg:>7.1f}")
    print("=" * 100)
    print()
    print("Random baseline at N=5000: R@1 = 0.02%, R@5 = 0.10%, R@10 = 0.20%, "
          "mean_rank = 2500")


if __name__ == "__main__":
    main()
