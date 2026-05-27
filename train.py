"""From-scratch CLIP training on cc12m with multi-GPU Accelerate.

Launch:
    CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 VERSION=3 accelerate launch train.py

All hyperparameters live in config.yaml. The VERSION env var (or --version
flag) chooses which runs/v{N}/ subdir the run lands in. An existing
runs/v{VERSION}/ is wiped at startup.

Multi-GPU notes:
    The contrastive loss is computed over the GLOBAL batch (all ranks'
    embeddings all-gathered), so each anchor sees world_size * batch - 1
    negatives instead of just its local batch. all_gather is not
    differentiable, so we replace the local rank's slice in the gathered
    tensor with the local tensor (OpenCLIP trick) — gradients flow back to
    the local model normally and DDP averages across ranks.
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import math
import shutil
import time
from pathlib import Path

import datetime

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

from dataset import build_loader, build_val_loader, BATCH_SIZE
from model import CLIPModel


# cc12m total - 200-shard val carveout. ~5040 pairs per shard on average, so
# we lose ~1M training pairs to validation.
PAIRS_PER_EPOCH = 10_968_539 - (200 * 5_040)
RUNS_ROOT       = Path(__file__).parent / "runs"
DEFAULT_CONFIG  = Path(__file__).parent / "config.yaml"
EVAL_BATCH      = 512                            # per-batch forward size during val
SIM_CHUNK       = 1024                           # query chunk size for streaming retrieval
AMP_DTYPE       = torch.bfloat16

REQUIRED_KEYS = (
    "version", "epochs", "warmup_steps", "base_lr", "backbone_lr_mult",
    "weight_decay", "grad_clip", "eval_every_frac", "log_every", "seed",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--version", type=int, default=None,
                    help="Override config.yaml's version field.")
    return ap.parse_args()


def load_config(path, version_override):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Override priority: --version > $VERSION env var > config.yaml
    if version_override is not None:
        cfg["version"] = version_override
    elif "VERSION" in os.environ:
        cfg["version"] = int(os.environ["VERSION"])

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"{path} missing keys: {missing}")

    # YAML can be loose about types; be explicit.
    cfg["version"]           = int(cfg["version"])
    cfg["epochs"]            = int(cfg["epochs"])
    cfg["warmup_steps"]      = int(cfg["warmup_steps"])
    cfg["base_lr"]           = float(cfg["base_lr"])
    cfg["backbone_lr_mult"]  = float(cfg["backbone_lr_mult"])
    cfg["weight_decay"]      = float(cfg["weight_decay"])
    cfg["grad_clip"]         = float(cfg["grad_clip"])
    cfg["eval_every_frac"]   = int(cfg["eval_every_frac"])
    cfg["log_every"]         = int(cfg["log_every"])
    cfg["seed"]              = int(cfg["seed"])
    return cfg


def make_logger(out_dir, is_main):
    if not is_main:
        return lambda msg: None
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")
    return log


def lr_schedule_factor(step, total_steps, warmup):
    """Returns a multiplier in [0, 1]: linear warmup, then cosine decay to 0.
    Applied per-param-group against that group's base_lr in the train loop."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def gather_features(local_img, local_txt, accelerator):
    """all_gather embeddings across ranks, keeping local-slice gradients."""
    if accelerator.num_processes == 1:
        return local_img, local_txt

    ws   = accelerator.num_processes
    rank = accelerator.process_index

    img_buf = [torch.zeros_like(local_img) for _ in range(ws)]
    txt_buf = [torch.zeros_like(local_txt) for _ in range(ws)]
    dist.all_gather(img_buf, local_img)
    dist.all_gather(txt_buf, local_txt)
    # all_gather is non-differentiable; splice the local tensor back in so
    # gradients flow into the local model.
    img_buf[rank] = local_img
    txt_buf[rank] = local_txt
    return torch.cat(img_buf, dim=0), torch.cat(txt_buf, dim=0)


def clip_loss_gathered(local_img, local_txt, logit_scale, accelerator):
    """Symmetric InfoNCE over the global batch.

    Returns (loss, R@1_i2t, R@1_t2i) where the R@1s are measured on local
    anchors against global candidates (so the printed in-batch number is
    against world_size * batch candidates, not just the local batch).
    """
    all_img, all_txt = gather_features(local_img, local_txt, accelerator)
    rank = accelerator.process_index
    bs   = local_img.shape[0]

    logits_i2t = logit_scale * local_img @ all_txt.T
    logits_t2i = logit_scale * local_txt @ all_img.T
    labels = torch.arange(bs, device=local_img.device) + rank * bs

    loss = 0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))

    with torch.no_grad():
        acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean()
        acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean()
    return loss, acc_i2t, acc_t2i


@torch.no_grad()
def streaming_metrics(img_emb, txt_emb, chunk=SIM_CHUNK):
    """R@{1,5,10} + mean_rank in both directions, computed in query-chunks.

    Avoids materialising the full NxN similarity matrix (which would be 4 TB
    at N=1M). Memory peaks at chunk x N x 4 bytes (4 GB at chunk=1024, N=1M).
    """
    n = img_emb.shape[0]
    device = img_emb.device
    labels = torch.arange(n, device=device)
    out = {}
    for direction, queries, keys in [
        ("i2t", img_emb, txt_emb),
        ("t2i", txt_emb, img_emb),
    ]:
        hits1 = hits5 = hits10 = 0
        rank_sum = 0.0
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            sims = queries[i:j] @ keys.T                        # [chunk, N]
            true_sim = sims.gather(1, labels[i:j].unsqueeze(1)) # [chunk, 1]
            rank = (sims > true_sim).sum(dim=1) + 1             # 1-indexed
            hits1    += (rank <= 1).sum().item()
            hits5    += (rank <= 5).sum().item()
            hits10   += (rank <= 10).sum().item()
            rank_sum += rank.float().sum().item()
        out[f"r@1_{direction}"]       = 100.0 * hits1  / n
        out[f"r@5_{direction}"]       = 100.0 * hits5  / n
        out[f"r@10_{direction}"]      = 100.0 * hits10 / n
        out[f"mean_rank_{direction}"] = rank_sum / n
    return out


@torch.no_grad()
def eval_on_val(model, val_loader, device, set_train_mode, log,
                max_pairs=None, progress_every=50):
    """Stream val_loader through the model, accumulate embeddings on GPU, then
    do chunked retrieval. Returns (metrics_dict, n_pairs_seen)."""
    model.eval()
    img_chunks, txt_chunks = [], []
    n_seen = 0
    last_logged = 0
    n_batches = 0

    with autocast(device_type="cuda", dtype=AMP_DTYPE):
        for batch in val_loader:
            imgs, ids, mask = batch
            imgs = imgs.to(device, non_blocking=True)
            ids  = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            ie, te, _ = model(imgs, ids, mask)
            img_chunks.append(ie.float())
            txt_chunks.append(te.float())
            n_seen += imgs.shape[0]
            n_batches += 1

            if n_seen - last_logged >= progress_every * imgs.shape[0]:
                log(f"    [eval] embedded {n_seen:,} pairs...")
                last_logged = n_seen

            if max_pairs is not None and n_seen >= max_pairs:
                break

    set_train_mode()

    img_emb = torch.cat(img_chunks, dim=0)
    txt_emb = torch.cat(txt_chunks, dim=0)
    if max_pairs is not None and img_emb.shape[0] > max_pairs:
        img_emb = img_emb[:max_pairs]
        txt_emb = txt_emb[:max_pairs]

    log(f"    [eval] embedded {img_emb.shape[0]:,} pairs total; computing retrieval metrics...")
    return streaming_metrics(img_emb, txt_emb), img_emb.shape[0]


def save_ckpt(path, step, model, optimizer):
    """Save the full model + optimizer state. Both backbones are trainable
    now, so we can't shortcut by skipping the text encoder anymore.
    Checkpoint size jumps from ~50MB to ~500MB+ because of this."""
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def manage_ckpts(new_path, new_score, best_path, best_score, last_path):
    """Keep only the best-so-far and the most-recent checkpoint on disk."""
    is_best = best_path is None or new_score > best_score

    to_delete = set()
    if last_path is not None and last_path != best_path:
        to_delete.add(last_path)
    if is_best and best_path is not None:
        to_delete.add(best_path)
    to_delete.discard(new_path)
    for p in to_delete:
        p.unlink(missing_ok=True)

    if is_best:
        best_path, best_score = new_path, new_score
    return best_path, best_score, new_path


def main():
    args = parse_args()
    cfg = load_config(args.config, args.version)

    version  = cfg["version"]
    out_dir  = RUNS_ROOT / f"v{version}"
    ckpt_dir = out_dir / "checkpoints"

    set_seed(cfg["seed"])
    # NCCL default timeout is 10 min. The 1M-pair eval on rank 0 takes ~30 min
    # while other ranks block at wait_for_everyone(), so the default would fire
    # the watchdog and kill the run. Bump to 2 h to give comfortable headroom.
    ddp_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_local_main_process
    device  = accelerator.device

    if is_main and out_dir.exists():
        shutil.rmtree(out_dir)
    accelerator.wait_for_everyone()

    log = make_logger(out_dir, is_main)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.snapshot.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    global_batch    = accelerator.num_processes * BATCH_SIZE
    steps_per_epoch = PAIRS_PER_EPOCH // global_batch
    total_steps     = cfg["epochs"] * steps_per_epoch
    eval_every      = max(1, steps_per_epoch // cfg["eval_every_frac"])

    log("=" * 78)
    log(f"From-scratch CLIP — version {version}")
    log("=" * 78)
    log(f"Config:            {args.config}")
    log(f"World size:        {accelerator.num_processes}  (rank {accelerator.process_index})")
    log(f"Per-rank batch:    {BATCH_SIZE}")
    log(f"Global batch:      {global_batch}")
    log(f"Epochs:            {cfg['epochs']}")
    log(f"Steps/epoch:       {steps_per_epoch:,}")
    log(f"Total steps:       {total_steps:,}")
    log(f"Eval every:        {eval_every:,} steps  ({cfg['eval_every_frac']}x per epoch)")
    log(f"Warmup:            {cfg['warmup_steps']} steps")
    log(f"Base LR (head):    {cfg['base_lr']}")
    log(f"Backbone LR mult:  {cfg['backbone_lr_mult']}  (backbone LR = {cfg['base_lr'] * cfg['backbone_lr_mult']:.2e})")
    log(f"Weight decay:      {cfg['weight_decay']}")
    log(f"Output:            {out_dir}")

    # Val loader is built lazily inside the eval block (rank 0 only) so we don't
    # spawn worker processes on the other ranks. Cap held-out at 1M pairs.
    val_max_pairs = 1_000_000

    log("Building train data loader...")
    loader = build_loader()
    loader_iter = iter(loader)

    log("Building model...")
    model = CLIPModel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trainable params: {n_trainable:,}")

    # 4-way param grouping: backbone vs head, cross weight_decay vs no_decay.
    #   - Backbones (image + text) get base_lr * backbone_lr_mult — gentler updates
    #     so we don't catastrophically forget the pretrained features.
    #   - Head params (projectors + log_scale) get the full base_lr.
    #   - No-decay set = 1D params (biases, LayerNorm), the learned temperature,
    #     and ViT-style positional / cls tokens (no-op for ResNet/EfficientNet
    #     but harmless to keep in the rule).
    NO_DECAY_KEYS     = ("log_scale", "pos_embed", "cls_token")
    BACKBONE_PREFIXES = ("image_backbone", "text_backbone")

    bb_decay, bb_no_decay, head_decay, head_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = any(n.startswith(prefix) for prefix in BACKBONE_PREFIXES)
        is_no_decay = p.ndim < 2 or any(k in n for k in NO_DECAY_KEYS)
        target = (bb_no_decay if is_no_decay else bb_decay) if is_backbone \
                 else (head_no_decay if is_no_decay else head_decay)
        target.append(p)

    backbone_lr = cfg["base_lr"] * cfg["backbone_lr_mult"]
    head_lr     = cfg["base_lr"]

    def _count(params): return sum(p.numel() for p in params)
    log(f"  backbone decay:    {len(bb_decay):>4} tensors, {_count(bb_decay):>12,} params  lr={backbone_lr}")
    log(f"  backbone no_decay: {len(bb_no_decay):>4} tensors, {_count(bb_no_decay):>12,} params  lr={backbone_lr}")
    log(f"  head decay:        {len(head_decay):>4} tensors, {_count(head_decay):>12,} params  lr={head_lr}")
    log(f"  head no_decay:     {len(head_no_decay):>4} tensors, {_count(head_no_decay):>12,} params  lr={head_lr}")

    # `base_lr` per group is stashed for the cosine schedule to pick up below.
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_decay,      "lr": backbone_lr, "weight_decay": cfg["weight_decay"], "base_lr": backbone_lr},
            {"params": bb_no_decay,   "lr": backbone_lr, "weight_decay": 0.0,                 "base_lr": backbone_lr},
            {"params": head_decay,    "lr": head_lr,     "weight_decay": cfg["weight_decay"], "base_lr": head_lr},
            {"params": head_no_decay, "lr": head_lr,     "weight_decay": 0.0,                 "base_lr": head_lr},
        ],
        betas=(0.9, 0.98), eps=1e-6,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    def set_train_mode():
        # Both backbones are trainable now — single .train() does the right thing.
        model.train()
    set_train_mode()

    log("Starting training loop...")
    t_start = time.time()
    t_log   = t_start
    step = 0
    running_loss = running_acc_i2t = running_acc_t2i = 0.0
    running_count = 0
    best_path = None
    best_score = float("-inf")
    last_path = None

    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images, input_ids, attention_mask = (x.to(device, non_blocking=True) for x in batch)

        # Same cosine schedule for both groups, just scaled by each group's base_lr.
        factor = lr_schedule_factor(step, total_steps, cfg["warmup_steps"])
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * factor

        with autocast(device_type="cuda", dtype=AMP_DTYPE):
            img_emb, txt_emb, logit_scale = model(images, input_ids, attention_mask)
            loss, acc_i2t, acc_t2i = clip_loss_gathered(img_emb, txt_emb, logit_scale, accelerator)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"],
        )
        optimizer.step()

        with torch.no_grad():
            accelerator.unwrap_model(model).log_scale.clamp_(max=math.log(100.0))

        running_loss    += loss.item()
        running_acc_i2t += acc_i2t.item()
        running_acc_t2i += acc_t2i.item()
        running_count   += 1

        if (step + 1) % cfg["log_every"] == 0:
            avg_loss = running_loss    / running_count
            avg_i2t  = running_acc_i2t / running_count
            avg_t2i  = running_acc_t2i / running_count
            running_loss = running_acc_i2t = running_acc_t2i = 0.0
            running_count = 0

            now = time.time()
            sps = (cfg["log_every"] * global_batch) / (now - t_log)
            t_log = now

            log(f"step {step+1:6d}/{total_steps} | "
                f"loss {avg_loss:.4f} | scale {logit_scale.item():.2f} | "
                f"lr {lr_now:.2e} | "
                f"R@1 i2t {avg_i2t*100:5.1f}% | R@1 t2i {avg_t2i*100:5.1f}% | "
                f"{sps:.0f} samples/s")

        if (step + 1) % eval_every == 0:
            accelerator.wait_for_everyone()
            if is_main:
                epoch_frac = (step + 1) / steps_per_epoch
                new_path = ckpt_dir / f"ckpt_step{step+1:08d}.pt"
                unwrapped = accelerator.unwrap_model(model)
                log(f"  [eval] step {step+1:,} (epoch {epoch_frac:.2f}/{cfg['epochs']}): "
                    f"saving ckpt + streaming up to {val_max_pairs:,} held-out pairs...")
                save_ckpt(new_path, step + 1, unwrapped, optimizer)

                # Build a fresh val loader each eval — webdataset iterators are
                # consumed, and we want a deterministic full pass each time.
                # Other ranks are blocked at wait_for_everyone() so rank 0 can
                # use lots of workers without contending for CPU.
                val_loader = build_val_loader(batch_size=EVAL_BATCH, num_workers=12)

                t_eval = time.time()
                m, n_val = eval_on_val(unwrapped, val_loader, device, set_train_mode,
                                       log, max_pairs=val_max_pairs)
                score = (m["r@1_i2t"] + m["r@1_t2i"]) / 2

                log(f"  eval @ step {step+1:,}  (epoch frac {epoch_frac:.2f}/{cfg['epochs']}, "
                    f"{time.time() - t_eval:.1f}s, n={n_val:,}):")
                log(f"    i2t R@1 {m['r@1_i2t']:5.2f}%  R@5 {m['r@5_i2t']:5.2f}%  "
                    f"R@10 {m['r@10_i2t']:5.2f}%  mean_rank {m['mean_rank_i2t']:7.1f}")
                log(f"    t2i R@1 {m['r@1_t2i']:5.2f}%  R@5 {m['r@5_t2i']:5.2f}%  "
                    f"R@10 {m['r@10_t2i']:5.2f}%  mean_rank {m['mean_rank_t2i']:7.1f}")
                log(f"    avg R@1 = {score:.3f}%  (prev best {best_score:.3f}%)")

                was_best = best_path is None or score > best_score
                best_path, best_score, last_path = manage_ckpts(
                    new_path, score, best_path, best_score, last_path,
                )
                log(f"    {'NEW BEST' if was_best else 'kept as last'}  "
                    f"-> best={best_path.name} ({best_score:.3f}%)  last={last_path.name}")
            accelerator.wait_for_everyone()

        step += 1

    total = time.time() - t_start
    log(f"\nTraining done in {total / 60:.1f} min ({total / 3600:.2f} h)")
    if best_path is not None:
        log(f"Best avg R@1: {best_score:.3f}%  -> {best_path.name}")


if __name__ == "__main__":
    main()
