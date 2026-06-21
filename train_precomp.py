"""Precomputed-embedding CLIP training on cc12m with multi-GPU Accelerate.

Launch:
    CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 VERSION=7 accelerate launch train_precomp.py

All hyperparameters live in config.yaml. The VERSION env var (or --version
flag) chooses runs/v{N}/. An existing runs/v{VERSION}/ is wiped at startup.

Pipeline:
    workers (decode + 256-crop → uint8 HWC)
        → PinnedPrefetcher (pinned-buf + side-stream async H2D)
        → GPUTrainTransform (RRC + flip + jitter + normalize on GPU)
        → CLIPPrecompModel (image fwd only; text emb comes from memmap)
        → DDP-gathered InfoNCE
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import datetime
import math
import shutil
import time
from pathlib import Path

import yaml
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

from dataset import build_loader, build_val_loader, BATCH_SIZE
from model import CLIPPrecompModel
from scripts.gpu_transforms import GPUTrainTransform, GPUValTransform
from scripts.prefetcher import PinnedPrefetcher
from scripts.embedding_lookup import load_embedding_table
from train_utils import train_steps, save_ckpt, manage_ckpts
from eval_utils import validate


PAIRS_PER_EPOCH = 10_968_539 - (200 * 5_040)        # cc12m train pairs
RUNS_ROOT       = Path(__file__).parent / "runs"
DEFAULT_CONFIG  = Path(__file__).parent / "config.yaml"
EMB_DIR         = Path("/mnt/md0/cc12m_qwen3_emb_4b_embeddings")
SHM_MMAP_PATH   = Path("/dev/shm/text_emb.mmap")
EVAL_BATCH      = 512
EVAL_MAX_PAIRS  = 1_000_000

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

def build_optimizer(model, cfg, log):
    """2-way × 2-way grouping: backbone vs head, decay vs no_decay.

    - image_backbone gets `base_lr * backbone_lr_mult` (gentler — don't
      catastrophically forget the pretrained features).
    - head = image_projector + text_projector + log_scale; gets `base_lr`.
    - no_decay = 1D params (biases, LayerNorm), log_scale, ViT-style tokens.
    """
    NO_DECAY_KEYS    = ("log_scale", "pos_embed", "cls_token")
    BACKBONE_PREFIX  = "image_backbone"

    bb_decay, bb_no_decay, head_decay, head_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = n.startswith(BACKBONE_PREFIX)
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

    return torch.optim.AdamW(
        [
            {"params": bb_decay,      "lr": backbone_lr, "weight_decay": cfg["weight_decay"], "base_lr": backbone_lr},
            {"params": bb_no_decay,   "lr": backbone_lr, "weight_decay": 0.0,                 "base_lr": backbone_lr},
            {"params": head_decay,    "lr": head_lr,     "weight_decay": cfg["weight_decay"], "base_lr": head_lr},
            {"params": head_no_decay, "lr": head_lr,     "weight_decay": 0.0,                 "base_lr": head_lr},
        ],
        betas=(0.9, 0.98), eps=1e-6,
    )

def main():
    args = parse_args()
    cfg  = load_config(args.config, args.version)

    version  = cfg["version"]
    out_dir  = RUNS_ROOT / f"v{version}"
    ckpt_dir = out_dir / "checkpoints"

    set_seed(cfg["seed"])
    # Bump NCCL timeout: 1M-pair eval on rank 0 takes ~5-10 min while other
    # ranks block at wait_for_everyone(); default 10-min watchdog would fire.
    ddp_kwargs  = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))
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

    # Step bookkeeping
    global_batch    = accelerator.num_processes * BATCH_SIZE
    steps_per_epoch = PAIRS_PER_EPOCH // global_batch
    total_steps     = cfg["epochs"] * steps_per_epoch
    eval_every      = max(1, steps_per_epoch // cfg["eval_every_frac"])
    cfg["per_rank_batch"] = BATCH_SIZE      # train_utils reads this

    log("=" * 78)
    log(f"Precomp CLIP — version {version}")
    log("=" * 78)
    log(f"World size:        {accelerator.num_processes}  (rank {accelerator.process_index})")
    log(f"Per-rank batch:    {BATCH_SIZE}")
    log(f"Global batch:      {global_batch}")
    log(f"Epochs:            {cfg['epochs']}")
    log(f"Steps/epoch:       {steps_per_epoch:,}")
    log(f"Total steps:       {total_steps:,}")
    log(f"Eval every:        {eval_every:,} steps  ({cfg['eval_every_frac']}x per epoch)")
    log(f"Warmup:            {cfg['warmup_steps']} steps")
    log(f"Base LR (head):    {cfg['base_lr']}")
    log(f"Backbone LR mult:  {cfg['backbone_lr_mult']}  (bb LR = {cfg['base_lr']*cfg['backbone_lr_mult']:.2e})")
    log(f"Output:            {out_dir}")

    # Data
    log("Building train loader + prefetcher + transforms...")
    loader     = build_loader()
    prefetcher = PinnedPrefetcher(loader, device)
    gpu_train  = GPUTrainTransform().to(device)
    gpu_val    = GPUValTransform().to(device)

    # Embedding table (rank 0 stages to /dev/shm, then all ranks mmap)
    log("Loading embedding table (will stage to /dev/shm on rank 0 if missing)...")
    text_table = load_embedding_table(EMB_DIR, SHM_MMAP_PATH, accelerator)
    log(f"  table shape: {text_table.shape}  dtype: {text_table.dtype}")

    # Model + optim
    log("Building model...")
    model = CLIPPrecompModel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trainable params: {n_trainable:,}")
    optimizer = build_optimizer(model, cfg, log)

    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()


    log("Starting training...")
    t_start        = time.time()
    log_state      = {"loss": 0.0, "i2t": 0.0, "t2i": 0.0, "n": 0,
                    "t_log": time.time()}
    prefetcher_iter = iter(prefetcher)

    step = 0
    best_path  = None
    best_score = float("-inf")
    last_path  = None

    while step < total_steps:
        # Train for the next eval-interval (or what remains of the run)
        n = min(eval_every, total_steps - step)
        step, prefetcher_iter = train_steps(
            n_steps=n, model=model, prefetcher_iter=prefetcher_iter,
            prefetcher=prefetcher, gpu_transform=gpu_train,
            text_table=text_table, optimizer=optimizer,
            accelerator=accelerator, cfg=cfg,
            start_step=step, total_steps=total_steps,
            log=log, log_state=log_state,
        )

        # Eval + ckpt management — rank 0 only; other ranks wait
        accelerator.wait_for_everyone()
        if is_main:
            epoch_frac = step / steps_per_epoch
            new_path   = ckpt_dir / f"ckpt_step{step:08d}.pt"
            log(f"  [eval] step {step:,} (epoch {epoch_frac:.2f}/{cfg['epochs']}): "
                f"saving ckpt + streaming up to {EVAL_MAX_PAIRS:,} held-out pairs...")
            unwrapped = accelerator.unwrap_model(model)
            save_ckpt(new_path, step, unwrapped, optimizer)

            val_loader = build_val_loader(batch_size=EVAL_BATCH, num_workers=12)
            t_eval = time.time()
            metrics, n_val = validate(
                model=unwrapped, val_loader=val_loader,
                gpu_val_transform=gpu_val, text_table=text_table,
                device=device, log=log, max_pairs=EVAL_MAX_PAIRS,
            )
            score = (metrics["r@1_i2t"] + metrics["r@1_t2i"]) / 2

            log(f"  eval @ step {step:,} ({time.time()-t_eval:.1f}s, n={n_val:,}):")
            log(f"    i2t R@1 {metrics['r@1_i2t']:5.2f}%  R@5 {metrics['r@5_i2t']:5.2f}%  "
                f"R@10 {metrics['r@10_i2t']:5.2f}%  mean_rank {metrics['mean_rank_i2t']:7.1f}")
            log(f"    t2i R@1 {metrics['r@1_t2i']:5.2f}%  R@5 {metrics['r@5_t2i']:5.2f}%  "
                f"R@10 {metrics['r@10_t2i']:5.2f}%  mean_rank {metrics['mean_rank_t2i']:7.1f}")
            log(f"    avg R@1 = {score:.3f}%  (prev best {best_score:.3f}%)")

            was_best = best_path is None or score > best_score
            best_path, best_score, last_path = manage_ckpts(
                new_path, score, best_path, best_score, last_path)
            log(f"    {'NEW BEST' if was_best else 'kept as last'}  "
                f"-> best={best_path.name} ({best_score:.3f}%)  last={last_path.name}")
        accelerator.wait_for_everyone()

    total = time.time() - t_start
    log(f"\nTraining done in {total/60:.1f} min ({total/3600:.2f} h)")
    if best_path is not None:
        log(f"Best avg R@1: {best_score:.3f}%  -> {best_path.name}")


if __name__ == "__main__":
    main()
