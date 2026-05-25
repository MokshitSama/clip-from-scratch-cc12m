"""Evaluate a checkpoint (or several) on the fixed 5000-pair held-out set.

For each checkpoint, reports R@{1,5,10} + mean rank in both directions.
Pass a directory to evaluate every ckpt_step*.pt inside.

    python evaluate.py runs/v3/checkpoints/ckpt_best.pt
    python evaluate.py runs/v3/checkpoints/  --csv runs/v3/eval.csv
"""
import argparse
import csv
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CLIPModel


def load_val(val_path):
    blob = torch.load(val_path, weights_only=False)
    return blob["images"], blob["input_ids"], blob["attention_mask"]


def load_model(ckpt_path, device):
    """Build a fresh CLIPModel; load saved parts. Frozen BERT comes from HF init."""
    model = CLIPModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)

    # Only the frozen text_backbone is allowed to be missing.
    bad = [k for k in missing if not k.startswith("text_backbone.")]
    if bad:
        raise RuntimeError(f"unexpected missing keys: {bad[:5]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in ckpt: {unexpected[:5]}")

    model.eval()
    return model, int(ckpt.get("step", -1))


@torch.no_grad()
def embed_all(model, images, input_ids, attn_mask, device, batch_size=256):
    n = images.shape[0]
    img_chunks, txt_chunks = [], []
    use_amp = device == "cuda"
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_amp else torch.amp.autocast(device_type="cpu", enabled=False))
    with ctx:
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            img_b = images[i:j].to(device, non_blocking=True)
            iid_b = input_ids[i:j].to(device, non_blocking=True)
            am_b  = attn_mask[i:j].to(device, non_blocking=True)
            img_emb, txt_emb, _ = model(img_b, iid_b, am_b)
            img_chunks.append(img_emb.float().cpu())
            txt_chunks.append(txt_emb.float().cpu())
    return torch.cat(img_chunks, dim=0), torch.cat(txt_chunks, dim=0)


def retrieval_metrics(sim):
    """R@k and mean/median rank for both directions on an NxN sim matrix."""
    n = sim.shape[0]
    labels = torch.arange(n, device=sim.device)
    out = {}
    for direction, scores in [("i2t", sim), ("t2i", sim.T)]:
        for k in (1, 5, 10):
            topk = scores.topk(k, dim=1).indices
            out[f"r@{k}_{direction}"] = (topk == labels[:, None]).any(dim=1).float().mean().item() * 100
        ranks = (scores.argsort(dim=1, descending=True) == labels[:, None]).float().argmax(dim=1) + 1
        out[f"mean_rank_{direction}"]   = ranks.float().mean().item()
        out[f"median_rank_{direction}"] = ranks.float().median().item()
    return out


def evaluate_one(ckpt_path, val_path, device, batch_size):
    images, input_ids, attn_mask = load_val(val_path)
    model, step = load_model(ckpt_path, device)
    img_emb, txt_emb = embed_all(model, images, input_ids, attn_mask, device, batch_size)
    sim = img_emb @ txt_emb.T
    m = retrieval_metrics(sim)
    m["step"] = step
    m["ckpt"] = str(ckpt_path)
    m["n"]    = img_emb.shape[0]
    return m


def print_metrics(m):
    print(f"  step {m['step']:>6} | n={m['n']}")
    print(f"  i2t  R@1 {m['r@1_i2t']:5.2f}%  R@5 {m['r@5_i2t']:5.2f}%  R@10 {m['r@10_i2t']:5.2f}%  "
          f"mean_rank {m['mean_rank_i2t']:6.2f}  median {m['median_rank_i2t']:.0f}")
    print(f"  t2i  R@1 {m['r@1_t2i']:5.2f}%  R@5 {m['r@5_t2i']:5.2f}%  R@10 {m['r@10_t2i']:5.2f}%  "
          f"mean_rank {m['mean_rank_t2i']:6.2f}  median {m['median_rank_t2i']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint files or a directory of them")
    ap.add_argument("--val", default=str(Path(__file__).parent / "val_set_cc12m_0020_5000.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    args = ap.parse_args()

    targets = []
    for c in args.ckpts:
        p = Path(c)
        if p.is_dir():
            targets.extend(sorted(p.glob("ckpt_step*.pt")))
        else:
            targets.append(p)
    if not targets:
        sys.exit("no checkpoints found")

    val_path = Path(args.val)
    print(f"val:    {val_path}")
    print(f"device: {args.device}")
    print(f"ckpts:  {len(targets)}")

    results = []
    for ckpt in targets:
        print(f"\n=== {ckpt.name} ===")
        m = evaluate_one(ckpt, val_path, args.device, args.batch)
        print_metrics(m)
        results.append(m)

    if args.csv:
        fields = ["ckpt", "step", "n",
                  "r@1_i2t", "r@5_i2t", "r@10_i2t", "mean_rank_i2t", "median_rank_i2t",
                  "r@1_t2i", "r@5_t2i", "r@10_t2i", "mean_rank_t2i", "median_rank_t2i"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m in results:
                w.writerow({k: m[k] for k in fields})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
