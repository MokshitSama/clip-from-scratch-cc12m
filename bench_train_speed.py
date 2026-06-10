"""Training-step throughput benchmark for the three text-encoder configs.

Pure compute benchmark: synthetic on-GPU tensors (no data loading), measuring
forward + backward + optimizer.step() wall time per iteration.

Configs:
  A  image-only            image backbone+proj trainable; text encoder NOT run
                           (txt embeddings are a fixed random constant). Isolates
                           the image-side training cost / loss ceiling.
  B  image + BGE(frozen)    image trainable; BGE forward runs but params frozen
                           (no backward into the 109M text params). text_projector
                           stays trainable.
  C  image + BGE(trainable) everything trainable — the current train.py setup.

Mirrors train.py's step: bf16 autocast, AdamW(betas=.9/.98, eps=1e-6, wd=0.1),
per-step grad-norm clip at 1.0. No seeding / determinism flags — same as training.
Runs each config both eager and torch.compile.

Run on a single free GPU, e.g. the idle 3090 (cuda:6 here):
    CUDA_VISIBLE_DEVICES=6 python bench_train_speed.py
    CUDA_VISIBLE_DEVICES=6 BATCH=256 CONFIGS=AB python bench_train_speed.py
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import torch
import torch.nn.functional as F
from torch.amp import autocast

from model import CLIPModel, clip_loss

DEVICE    = "cuda"
AMP       = torch.bfloat16
BATCH     = int(os.environ.get("BATCH", "128"))
WARMUP    = int(os.environ.get("WARMUP", "8"))
ITERS     = int(os.environ.get("ITERS", "30"))
SEQ_LEN   = 64
GRAD_CLIP = 1.0          # matches config.yaml; train.py clips grad norm every step


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def build(config):
    """Return (model, trainable_params, run_step_fn) for a config."""
    model = CLIPModel().to(DEVICE)

    if config == "A":          # image only — text encoder never runs
        set_requires_grad(model.text_backbone, False)
        set_requires_grad(model.text_projector, False)
    elif config == "B":        # image trainable, BGE frozen (proj still trains)
        set_requires_grad(model.text_backbone, False)
    elif config == "C":        # everything trainable
        pass

    model.train()
    return model


def make_fwd(model, config, fixed_txt):
    """Forward-only callable (the part torch.compile wraps). Autocast + loss +
    backward stay outside in one_iter so we compile the model graph, not the
    optimizer."""
    if config == "A":
        def fwd(images, ids, mask):
            img_emb = F.normalize(model.encode_image(images), dim=-1)
            scale   = model.log_scale.exp().clamp(max=100.0)
            return img_emb, fixed_txt, scale          # fixed_txt: constant, no grad
    else:
        def fwd(images, ids, mask):
            return model(images, ids, mask)
    return fwd


def bench(config, use_compile):
    import torch._dynamo
    torch._dynamo.reset()                 # clear compiled-graph cache between runs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build(config)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)

    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.98), eps=1e-6,
                                  weight_decay=0.1)

    images = torch.randn(BATCH, 3, 224, 224, device=DEVICE)
    ids    = torch.randint(0, 30000, (BATCH, SEQ_LEN), device=DEVICE)
    mask   = torch.ones(BATCH, SEQ_LEN, dtype=torch.long, device=DEVICE)
    # Fixed normalized random text embeddings for config A (512-d to match proj out).
    fixed_txt = F.normalize(torch.randn(BATCH, 512, device=DEVICE), dim=-1)

    fwd = make_fwd(model, config, fixed_txt)
    if use_compile:
        fwd = torch.compile(fwd)

    def one_iter():
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=AMP):
            img_emb, txt_emb, scale = fwd(images, ids, mask)
            loss = clip_loss(img_emb, txt_emb, scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)   # train.py clips every step
        optimizer.step()
        return loss

    t_compile0 = time.time()
    for _ in range(WARMUP):           # first iters trigger compilation (not timed)
        one_iter()
    torch.cuda.synchronize()
    warmup_s = time.time() - t_compile0

    t0 = time.time()
    for _ in range(ITERS):
        one_iter()
    torch.cuda.synchronize()
    dt = time.time() - t0

    sps      = BATCH * ITERS / dt
    ms_iter  = 1000 * dt / ITERS
    peak_gb  = torch.cuda.max_memory_allocated() / 1e9

    del model, optimizer, trainable, fwd
    torch.cuda.empty_cache()
    return dict(config=config, compile=use_compile, n_train=n_train,
                sps=sps, ms_iter=ms_iter, peak_gb=peak_gb, warmup_s=warmup_s)


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  |  batch={BATCH}  "
          f"seq_len={SEQ_LEN}  amp=bf16  grad_clip={GRAD_CLIP}")
    print(f"warmup={WARMUP} iters, timed={ITERS} iters\n")

    labels = {
        "A": "image-only (trainable)",
        "B": "image(train) + BGE(frozen)",
        "C": "image + BGE (both trainable)",
    }
    configs = tuple(os.environ.get("CONFIGS", "ABC"))

    results = {}                          # (cfg, use_compile) -> row
    for cfg in configs:
        for use_compile in (False, True):
            mode = "compile" if use_compile else "eager  "
            try:
                r = bench(cfg, use_compile)
                results[(cfg, use_compile)] = r
                print(f"[{cfg}] {labels[cfg]:<30} {mode}  "
                      f"{r['sps']:7.1f} samples/s  |  {r['ms_iter']:7.1f} ms/iter  |  "
                      f"peak {r['peak_gb']:5.2f} GB  |  warmup {r['warmup_s']:5.1f}s")
            except RuntimeError as e:
                print(f"[{cfg}] {labels[cfg]:<30} {mode}  FAILED: {str(e)[:70]}")
                torch.cuda.empty_cache()

    print("\neager -> compile (per config):")
    for cfg in configs:
        e = results.get((cfg, False))
        c = results.get((cfg, True))
        if e and c:
            print(f"  [{cfg}] {labels[cfg]:<30} "
                  f"{e['sps']:7.1f} -> {c['sps']:7.1f} samples/s   {c['sps']/e['sps']:.2f}x")
