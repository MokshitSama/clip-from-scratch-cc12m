"""CLIP training with precomputed text embeddings, multi-GPU via Accelerate.

What makes this trainer fast (each item was added after profiling showed it
mattered — the [prof] log line breaks every step into phases so the biggest
cost is always visible):

  - no text tower: captions are frozen embeddings read from a RAM-staged
    table (dataset_precomp.py), so the only trained network is the image side
  - uint8 image transport + pinned staging buffers (gpu_transforms.py)
  - augmentation on the GPU, written as plain tensor ops
  - torch.compile on the train forward, cudnn.benchmark, TF32, fused AdamW

Reference numbers on 6x RTX 5090, batch 512/GPU: ~380ms/step, ~8,000 images/s.
Same machine before this work: ~2.5-3.5k images/s.

Launch (GPUs 0-5 are the 5090s on this box):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 VERSION=11 accelerate launch train_precomp.py

VERSION picks the output folder runs/v{N}/ (wiped at startup if it exists).
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import datetime
import math
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

import dataset_precomp as D
from dataset_precomp import build_loader, build_val_loader, BATCH_SIZE
from gpu_transforms import GPUTransform, PinnedPrefetcher
from model_precomp import CLIPPrecomp


PAIRS_PER_EPOCH = 10_968_539 - (200 * 5_040)   # all of cc12m minus the 200 held-out shards
RUNS_ROOT       = Path(__file__).parent / "runs"
DEFAULT_CONFIG  = Path(__file__).parent / "config.yaml"
EVAL_BATCH      = 512
SIM_CHUNK       = 512                          # rows per chunk in eval retrieval (see streaming_metrics)
AMP_DTYPE       = torch.bfloat16

REQUIRED_KEYS = (
    "version", "epochs", "warmup_steps", "base_lr", "backbone_lr_mult",
    "weight_decay", "grad_clip", "eval_every_frac", "log_every", "seed",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--version", type=int, default=None)
    return ap.parse_args()


def load_config(path, version_override):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if version_override is not None:
        cfg["version"] = version_override
    elif "VERSION" in os.environ:
        cfg["version"] = int(os.environ["VERSION"])
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"{path} missing keys: {missing}")
    # YAML is loose about types (5e-4 can parse as a string); be explicit.
    cfg["version"]          = int(cfg["version"])
    cfg["epochs"]           = int(cfg["epochs"])
    cfg["warmup_steps"]     = int(cfg["warmup_steps"])
    cfg["base_lr"]          = float(cfg["base_lr"])
    cfg["backbone_lr_mult"] = float(cfg["backbone_lr_mult"])
    cfg["weight_decay"]     = float(cfg["weight_decay"])
    cfg["grad_clip"]        = float(cfg["grad_clip"])
    cfg["eval_every_frac"]  = int(cfg["eval_every_frac"])
    cfg["log_every"]        = int(cfg["log_every"])
    cfg["seed"]             = int(cfg["seed"])
    cfg["profile"]          = bool(cfg.get("profile", True))   # optional key, on by default
    return cfg


def make_logger(out_dir, is_main):
    # Only rank 0 logs; the other ranks get a no-op so the code can call
    # log() unconditionally.
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


class StepProfiler:
    """Splits each training step into named phases and reports ms and % per
    phase every log window. This is the instrument the whole optimization
    effort was driven by: fix the biggest number, measure again.

    GPU work is asynchronous — python "finishes" a line long before the GPU
    does — so each phase boundary calls cuda.synchronize() to make sure work
    is billed to the phase that launched it. The sync overhead is negligible
    next to a ~380ms step (and the loop already syncs once per step via
    loss.item()).

    Reading the report:
      - data+h2d     waiting for the prefetcher + the copy to the GPU. High
                     means the data pipeline can't keep up.
      - loss+gather  contains the cross-GPU all_gather, which no rank can
                     pass until every rank arrives. Wait for the slowest rank
                     lands here — so a data stall on ANY rank shows up in
                     everyone else's loss+gather. (Measured alone, the
                     collective itself costs ~4ms.)
      - backward     includes DDP gradient averaging between GPUs.
    Profiling runs on rank 0 only.
    """
    def __init__(self, enabled):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.acc = defaultdict(float)
        self.steps = 0

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.acc[name] += time.perf_counter() - t0

    def step_done(self):
        self.steps += 1

    def report(self):
        """ms/step and % per phase since the last report."""
        if not self.enabled or self.steps == 0:
            return None
        total_ms = sum(self.acc.values()) / self.steps * 1000
        parts = []
        for name, t in sorted(self.acc.items(), key=lambda kv: -kv[1]):
            ms = t / self.steps * 1000
            parts.append(f"{name} {ms:7.1f}ms ({100*ms/max(total_ms,1e-9):4.1f}%)")
        msg = f"[prof] {total_ms:7.1f} ms/step | " + " | ".join(parts)
        self.reset()
        return msg


def lr_schedule_factor(step, total_steps, warmup):
    """Linear warmup, then cosine decay to zero. Returned as a 0..1 factor so
    each param group can scale its own base LR by it."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def gather_features(local_img, local_txt, accelerator):
    """Collect every GPU's embeddings so the loss can use all of them as
    negatives. all_gather gives back plain tensors with no gradient history,
    so each rank splices its own live tensor back into the gathered list —
    gradients then flow into the local model as usual and DDP averages them
    across ranks (the OpenCLIP trick)."""
    if accelerator.num_processes == 1:
        return local_img, local_txt
    ws   = accelerator.num_processes
    rank = accelerator.process_index
    img_buf = [torch.zeros_like(local_img) for _ in range(ws)]
    txt_buf = [torch.zeros_like(local_txt) for _ in range(ws)]
    dist.all_gather(img_buf, local_img)
    dist.all_gather(txt_buf, local_txt)
    img_buf[rank] = local_img
    txt_buf[rank] = local_txt
    return torch.cat(img_buf, dim=0), torch.cat(txt_buf, dim=0)


def clip_loss_gathered(local_img, local_txt, logit_scale, accelerator):
    """Contrastive loss over the GLOBAL batch: each image must pick out its
    caption from world_size * batch candidates, not just its own GPU's worth.
    More negatives per step is the main reason to gather at all.

    Also returns in-batch R@1 both directions: of this rank's queries, the
    fraction whose true partner scored highest among all global candidates.
    A health signal for the log, not a real retrieval metric — the candidate
    pool here is ~3k, the eval pool is 1M.
    """
    all_img, all_txt = gather_features(local_img, local_txt, accelerator)
    rank = accelerator.process_index
    bs   = local_img.shape[0]
    logits_i2t = logit_scale * local_img @ all_txt.T
    logits_t2i = logit_scale * local_txt @ all_img.T
    # This rank's pairs sit at rows rank*bs .. rank*bs+bs-1 of the gathered set.
    labels = torch.arange(bs, device=local_img.device) + rank * bs
    loss = 0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))
    with torch.no_grad():
        acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean()
        acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean()
    return loss, acc_i2t, acc_t2i


@torch.no_grad()
def streaming_metrics(img_emb, txt_emb, chunk=SIM_CHUNK):
    """R@{1,5,10} and mean rank, both directions, on N pairs.

    The full N x N similarity matrix would be 4TB at N=1M, so queries are
    processed in chunks: score one slice against all N keys, count how many
    candidates beat the true match, move on. Peak memory is chunk x N.

    empty_cache() first: training fragments the allocator's memory, and the
    big chunk allocations here can fail even when enough total memory is
    free (that was issue #6 — a silent OOM on the second eval).
    """
    torch.cuda.empty_cache()
    n = img_emb.shape[0]
    device = img_emb.device
    labels = torch.arange(n, device=device)
    out = {}
    for direction, queries, keys in [("i2t", img_emb, txt_emb), ("t2i", txt_emb, img_emb)]:
        hits1 = hits5 = hits10 = 0
        rank_sum = 0.0
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            sims = queries[i:j] @ keys.T
            true_sim = sims.gather(1, labels[i:j].unsqueeze(1))
            # Strict '>' means ties don't count against the true match.
            rank = (sims > true_sim).sum(dim=1) + 1
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
def eval_on_val(model, val_loader, device, gpu_tf, set_train_mode, log, max_pairs=None, progress_every=50):
    """Embed up to max_pairs held-out pairs, then run chunked retrieval.
    Runs on rank 0 while the other ranks wait at a barrier."""
    model.eval()
    img_chunks, txt_chunks = [], []
    n_seen = last_logged = 0
    with autocast(device_type="cuda", dtype=AMP_DTYPE):
        for imgs, text_emb in val_loader:
            imgs = gpu_tf.val(imgs.to(device, non_blocking=True))
            text_emb = text_emb.to(device, non_blocking=True)
            ie, te, _ = model(imgs, text_emb)
            img_chunks.append(ie.float())
            txt_chunks.append(te.float())
            n_seen += imgs.shape[0]
            if n_seen - last_logged >= progress_every * imgs.shape[0]:
                log(f"    [eval] embedded {n_seen:,} pairs...")
                last_logged = n_seen
            if max_pairs is not None and n_seen >= max_pairs:
                break
    set_train_mode()
    img_emb = torch.cat(img_chunks, dim=0)
    txt_emb = torch.cat(txt_chunks, dim=0)
    if max_pairs is not None and img_emb.shape[0] > max_pairs:
        img_emb, txt_emb = img_emb[:max_pairs], txt_emb[:max_pairs]
    del img_chunks, txt_chunks
    torch.cuda.empty_cache()
    log(f"    [eval] embedded {img_emb.shape[0]:,} pairs total; computing retrieval metrics...")
    metrics = streaming_metrics(img_emb, txt_emb)
    n_pairs = img_emb.shape[0]
    del img_emb, txt_emb
    torch.cuda.empty_cache()
    return metrics, n_pairs


def save_ckpt(path, step, model, optimizer):
    torch.save({"step": step, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()}, path)


def manage_ckpts(new_path, new_score, best_path, best_score, last_path):
    """Keep exactly two checkpoints on disk: best so far, and most recent."""
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
    # We don't need bit-exact reproducibility, so trade it for speed:
    # cudnn.benchmark lets cuDNN time its conv algorithms and keep the
    # fastest; TF32 uses faster matmul hardware at slightly lower precision.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Eval on rank 0 takes longer than NCCL's default 10-minute patience while
    # the other ranks sit at a barrier; don't let the watchdog kill the run.
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

    # Put the embedding table in RAM before any worker touches it. Rank 0
    # does the copy; everyone else waits, then resolves the same path.
    if is_main:
        emb_dir = D.stage_embeddings_to_shm()
        log(f"Embeddings staged at: {emb_dir}")
    accelerator.wait_for_everyone()
    emb_dir  = D.stage_embeddings_to_shm()
    text_dim = D.text_emb_dim(emb_dir)

    log("=" * 78)
    log(f"Precomputed-text CLIP — version {version}")
    log("=" * 78)
    log(f"Embeddings:        {emb_dir}  (dim {text_dim})")
    log(f"World size:        {accelerator.num_processes}  (rank {accelerator.process_index})")
    log(f"Per-rank batch:    {BATCH_SIZE}   Global batch: {global_batch}")
    log(f"Epochs:            {cfg['epochs']}   Steps/epoch: {steps_per_epoch:,}   Total: {total_steps:,}")
    log(f"Eval every:        {eval_every:,} steps  ({cfg['eval_every_frac']}x/epoch)")
    log(f"Output:            {out_dir}")

    val_max_pairs = 1_000_000

    log("Building train data loader...")
    loader = build_loader(emb_dir=emb_dir)
    prefetcher = PinnedPrefetcher(
        loader, device,
        batch_spec=[((BATCH_SIZE, D.TRAIN_CANVAS, D.TRAIN_CANVAS, 3), torch.uint8),
                    ((BATCH_SIZE, text_dim), torch.float16)],
        n_slots=8,                      # ~830MB pinned per rank, ~3.5s of cushion
        log=log,
    )

    log("Building model...")
    model = CLIPPrecomp(text_emb_dim=text_dim)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trainable params: {n_trainable:,}")

    # Two split axes for the optimizer. Backbone vs head: the pretrained
    # image backbone gets a gentler LR so its ImageNet features aren't
    # trampled early, while the from-scratch projectors learn at full speed.
    # Decay vs no-decay: weight decay shrinks weights toward zero, which is
    # regularization for matrices but harmful for biases, norm gains and the
    # temperature — those go in the no-decay groups.
    NO_DECAY_KEYS     = ("log_scale", "pos_embed", "cls_token")
    BACKBONE_PREFIXES = ("image_backbone",)
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
    log(f"  backbone params: {sum(p.numel() for p in bb_decay+bb_no_decay):,}  lr={backbone_lr}")
    log(f"  head params:     {sum(p.numel() for p in head_decay+head_no_decay):,}  lr={head_lr}")

    # fused=True runs the whole AdamW update in one GPU kernel instead of
    # looping over parameter tensors. base_lr is stashed per group so the
    # schedule below can rescale each group from its own peak.
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_decay,      "lr": backbone_lr, "weight_decay": cfg["weight_decay"], "base_lr": backbone_lr},
            {"params": bb_no_decay,   "lr": backbone_lr, "weight_decay": 0.0,                 "base_lr": backbone_lr},
            {"params": head_decay,    "lr": head_lr,     "weight_decay": cfg["weight_decay"], "base_lr": head_lr},
            {"params": head_no_decay, "lr": head_lr,     "weight_decay": 0.0,                 "base_lr": head_lr},
        ],
        betas=(0.9, 0.98), eps=1e-6, fused=True,
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    # Compile only the training forward. Eval and checkpointing use the plain
    # module (same weights underneath): eval batches vary in size, which
    # would trigger recompiles, and the plain module's state_dict has clean
    # key names.
    compiled = torch.compile(model)
    gpu_tf = GPUTransform(device)

    def set_train_mode():
        model.train()
    set_train_mode()

    log("Starting training loop (cudnn.benchmark on, torch.compile on, fused AdamW)...")
    prof = StepProfiler(enabled=is_main and cfg["profile"])
    t_start = t_log = time.time()
    step = 0
    running_loss = running_acc_i2t = running_acc_t2i = 0.0
    running_count = 0
    best_path = None
    best_score = float("-inf")
    last_path = None

    while step < total_steps:
        with prof.phase("data+h2d"):
            images, text_emb = prefetcher.next()

        with prof.phase("gpu_aug"):
            images = gpu_tf.train(images)

        factor = lr_schedule_factor(step, total_steps, cfg["warmup_steps"])
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * factor

        with autocast(device_type="cuda", dtype=AMP_DTYPE):
            with prof.phase("forward"):
                img_emb, txt_emb, logit_scale = compiled(images, text_emb)
            with prof.phase("loss+gather"):
                loss, acc_i2t, acc_t2i = clip_loss_gathered(img_emb, txt_emb, logit_scale, accelerator)

        with prof.phase("backward"):
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

        with prof.phase("optim"):
            accelerator.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            optimizer.step()
            with torch.no_grad():
                accelerator.unwrap_model(model).log_scale.clamp_(max=math.log(100.0))

        with prof.phase("metrics"):
            running_loss    += loss.item()
            running_acc_i2t += acc_i2t.item()
            running_acc_t2i += acc_t2i.item()
            running_count   += 1
        prof.step_done()

        if (step + 1) % cfg["log_every"] == 0:
            avg_loss = running_loss    / running_count
            avg_i2t  = running_acc_i2t / running_count
            avg_t2i  = running_acc_t2i / running_count
            running_loss = running_acc_i2t = running_acc_t2i = 0.0
            running_count = 0
            now = time.time()
            sps = (cfg["log_every"] * global_batch) / (now - t_log)
            t_log = now
            head_lr_now = optimizer.param_groups[2]["lr"]
            bb_lr_now   = optimizer.param_groups[0]["lr"]
            log(f"step {step+1:6d}/{total_steps} | loss {avg_loss:.4f} | scale {logit_scale.item():.2f} | "
                f"lr {head_lr_now:.2e}/{bb_lr_now:.2e} (head/bb) | "
                f"R@1 i2t {avg_i2t*100:5.1f}% | R@1 t2i {avg_t2i*100:5.1f}% | {sps:.0f} samples/s")
            if cfg["profile"]:
                prof_msg = prof.report()
                if prof_msg:
                    log(prof_msg)
                # Each rank reports how long it sat waiting for data this
                # window. This is the line that finds stragglers: a stall on
                # any single rank is invisible in rank 0's [prof] but shows
                # up here by name. All ranks must execute the all_gather.
                wait_ms = torch.tensor([prefetcher.pop_wait_ms() / cfg["log_every"]], device=device)
                if accelerator.num_processes > 1:
                    waits = [torch.zeros_like(wait_ms) for _ in range(accelerator.num_processes)]
                    dist.all_gather(waits, wait_ms)
                    log("[data-wait/rank] " + "  ".join(f"r{i} {w.item():6.1f}" for i, w in enumerate(waits)) + "  ms/step")
                else:
                    log(f"[data-wait/rank] r0 {wait_ms.item():6.1f} ms/step")

        if (step + 1) % eval_every == 0:
            accelerator.wait_for_everyone()
            if is_main:
                epoch_frac = (step + 1) / steps_per_epoch
                new_path = ckpt_dir / f"ckpt_step{step+1:08d}.pt"
                unwrapped = accelerator.unwrap_model(model)
                log(f"  [eval] step {step+1:,} (epoch {epoch_frac:.2f}/{cfg['epochs']}): saving + streaming val...")
                save_ckpt(new_path, step + 1, unwrapped, optimizer)
                val_loader = build_val_loader(batch_size=EVAL_BATCH, num_workers=16, emb_dir=emb_dir)
                t_eval = time.time()
                m, n_val = eval_on_val(unwrapped, val_loader, device, gpu_tf, set_train_mode, log, max_pairs=val_max_pairs)
                score = (m["r@1_i2t"] + m["r@1_t2i"]) / 2
                log(f"  eval @ step {step+1:,} ({time.time()-t_eval:.1f}s, n={n_val:,}):")
                log(f"    i2t R@1 {m['r@1_i2t']:5.2f}%  R@5 {m['r@5_i2t']:5.2f}%  R@10 {m['r@10_i2t']:5.2f}%  mean_rank {m['mean_rank_i2t']:7.1f}")
                log(f"    t2i R@1 {m['r@1_t2i']:5.2f}%  R@5 {m['r@5_t2i']:5.2f}%  R@10 {m['r@10_t2i']:5.2f}%  mean_rank {m['mean_rank_t2i']:7.1f}")
                log(f"    avg R@1 = {score:.3f}%  (prev best {best_score:.3f}%)")
                was_best = best_path is None or score > best_score
                best_path, best_score, last_path = manage_ckpts(new_path, score, best_path, best_score, last_path)
                log(f"    {'NEW BEST' if was_best else 'kept as last'}  -> best={best_path.name} ({best_score:.3f}%)")
            accelerator.wait_for_everyone()

        step += 1

    total = time.time() - t_start
    log(f"\nTraining done in {total/60:.1f} min ({total/3600:.2f} h)")
    if best_path is not None:
        log(f"Best avg R@1: {best_score:.3f}%  -> {best_path.name}")


if __name__ == "__main__":
    main()
