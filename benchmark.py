"""Benchmark v6 + v7 against OpenAI CLIP and OpenCLIP cc12m.

Uses the cached 5000-pair set from cc12m shard 0020. All models score on
the SAME 5000 images / 5000 captions — only the preprocessing differs
(each model's own normalization + tokenizer / text encoder).

v6 has a trainable BGE-base text encoder baked in. v7 (precomp) loads
text embeddings on demand by re-encoding the val captions through the
Qwen3-Embedding-4B that produced the training table — that takes ~2 min
of Qwen3 load + encode, then Qwen3 is freed before any other model is
loaded.

Caveats worth knowing:
- Shard 0020 is in our held-out set. For OpenAI CLIP, unseen. For
  OpenCLIP cc12m it could be in their training data (they used all of
  cc12m). The advantage there is real but bounded.
- Different architectures + param counts. We're comparing recipes more
  than apples-to-apples.
- 5000 candidates is the CLIP-paper standard for COCO-style eval.

Run with:
    CUDA_VISIBLE_DEVICES=4 python benchmark.py   # 3090
    CUDA_VISIBLE_DEVICES=0 python benchmark.py   # a 5090 (faster)
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CLIPModel, CLIPPrecompModel


DEVICE   = "cuda"
VAL_PATH = Path(__file__).parent / "val_set_cc12m_0020_5000.pt"
V6_DIR   = Path(__file__).parent / "runs" / "v6" / "checkpoints"
V7_DIR   = Path(__file__).parent / "runs" / "v7" / "checkpoints"
QWEN_MODEL = "Qwen/Qwen3-Embedding-4B"
QWEN_MAX_LEN = 128


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


def find_latest_ckpt(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {ckpt_dir}")
    return ckpts[-1]


@torch.no_grad()
def embed_v6(model, val, bs=64):
    """v6: CLIPModel forward with ImageNet-normalized images + BGE tokens."""
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
def embed_v7(model, val, text_emb_qwen, bs=64):
    """v7: CLIPPrecompModel forward with ImageNet-normalized images +
    precomputed Qwen3-Embedding-4B text embeddings (2560-dim)."""
    images = val["images"].to(DEVICE)
    img_out, txt_out = [], []
    for i in range(0, images.shape[0], bs):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ie, te, _ = model(images[i:i+bs], text_emb_qwen[i:i+bs])
        img_out.append(ie.float())
        txt_out.append(te.float())
    return torch.cat(img_out), torch.cat(txt_out)


@torch.no_grad()
def embed_openclip(model, tokenizer, val, bs=32, max_text_len=77):
    """OpenCLIP path: re-normalize images to CLIP stats, re-tokenize captions."""
    imgs = renormalize_imagenet_to_clip(val["images"].to(DEVICE))
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


def encode_captions_qwen(captions):
    """Run Qwen3-Embedding-4B on the val captions to produce 2560-dim
    fp32 normalized text embeddings. Frees Qwen3 before returning."""
    from sentence_transformers import SentenceTransformer
    print(f"  loading {QWEN_MODEL} (sdpa)...")
    qwen = SentenceTransformer(
        QWEN_MODEL, device=DEVICE,
        model_kwargs={"torch_dtype": torch.bfloat16,
                      "attn_implementation": "sdpa"},
    )
    qwen.max_seq_length = QWEN_MAX_LEN
    t0 = time.time()
    text_emb = qwen.encode(
        captions, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    print(f"  encoded {len(captions)} captions in {time.time()-t0:.1f}s, "
          f"shape={text_emb.shape}")
    del qwen
    torch.cuda.empty_cache()
    return torch.from_numpy(text_emb).to(DEVICE)


def retrieval_metrics(img_emb, txt_emb):
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
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using device: {name}  ({total_gb:.1f} GB)\n")

    print(f"Loading val set: {VAL_PATH.name}")
    val = torch.load(VAL_PATH, weights_only=False)
    n = val["images"].shape[0]
    print(f"  {n} pairs\n")

    results = {}
    sizes   = {}

    # --- v7 first: encode captions through Qwen3, then run v7 forward, free both ---
    print("Encoding val captions through Qwen3-Embedding-4B (for v7)...")
    text_emb_qwen = encode_captions_qwen(val["captions"])

    v7_ckpt = find_latest_ckpt(V7_DIR)
    print(f"\nLoading v7 ({v7_ckpt.name})...")
    v7 = CLIPPrecompModel().to(DEVICE)
    ckpt = torch.load(v7_ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = v7.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  step: {ckpt.get('step')}  missing: {len(missing)}  unexpected: {len(unexpected)}")
    v7.eval()
    sizes["v7 (ours)"] = sum(p.numel() for p in v7.parameters())

    print("Embedding with v7...")
    t0 = time.time()
    img, txt = embed_v7(v7, val, text_emb_qwen)
    print(f"  done in {time.time()-t0:.1f}s")
    results["v7 (ours)"] = retrieval_metrics(img, txt)
    del v7, img, txt, text_emb_qwen
    torch.cuda.empty_cache()

    # --- v6 ---
    v6_ckpt = find_latest_ckpt(V6_DIR)
    print(f"\nLoading v6 ({v6_ckpt.name})...")
    v6 = CLIPModel().to(DEVICE)
    ckpt = torch.load(v6_ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = v6.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  step: {ckpt.get('step')}  missing: {len(missing)}  unexpected: {len(unexpected)}")
    v6.eval()
    sizes["v6 (ours)"] = sum(p.numel() for p in v6.parameters())

    print("Embedding with v6...")
    t0 = time.time()
    img, txt = embed_v6(v6, val)
    print(f"  done in {time.time()-t0:.1f}s")
    results["v6 (ours)"] = retrieval_metrics(img, txt)
    del v6, img, txt
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
        print(f"  done in {time.time()-t0:.1f}s")
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
