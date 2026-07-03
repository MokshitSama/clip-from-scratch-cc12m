# CLIP from scratch on cc12m

A small implementation of [CLIP](https://arxiv.org/abs/2103.00020) trained from scratch on
[CC12M](https://github.com/google-research-datasets/conceptual-12m) (~11M image–caption pairs).
The image encoder is a pretrained EfficientNet B1 (current default; swappable via the
`image_backbone` arg in `model.py`). The text encoder is a frozen, pretrained sentence-
embedding model — `BAAI/bge-base-en-v1.5` by default, a modern retrieval-trained model
that drops in for BERT but with much stronger semantic embeddings. Two small MLP
projectors map both into a shared 512-dim space, and a learned temperature scales the
cosine-similarity logits.

Multi-GPU with 🤗 Accelerate, using the OpenCLIP "gather features" trick so that the
contrastive loss is computed over the full *global* batch rather than each rank's local
slice — i.e. on N GPUs at batch B per rank, every anchor sees N·B − 1 negatives, not B − 1.

> **TL;DR — current state.** Three versions cover the story: v6 is the trainable-text
> InfoNCE baseline (4.53 % R@1 on 1M-pair held-out val, 22.4 h on 6×5090); v7 froze the
> text encoder and rebuilt the pipeline for a 2× throughput win but lost 2 R@1
> (2.47 %, 12.9 h); v8 is running now with SigLIP loss + K=64 image batch + 1024
> random text negatives per step and has already cleared v6 (5.15 % R@1 at epoch 9
> of 20, still climbing). Full walkthrough in
> [Performance pipeline](#performance-pipeline-v6-speedups) →
> [v6 vs v7](#v6-vs-v7--what-the-pipeline-actually-bought-us) →
> [v8 — SigLIP](#v8--siglip-recovers-the-quality-v7-lost-branch).

## Architecture

```
image (224x224) -> EfficientNet B1 (ImageNet-pretrained, trainable) -> 1280 -> MLP -> 512 ─┐
                                                                                            ├─ cosine sim
text  (raw caption) -> Qwen3-Embedding-4B (frozen, precomputed to /dev/shm) -> 2560 -> MLP -> 512 ─┘
                                                                                            ↑
                                                                                 v6/v7: log_scale (init 0)
                                                                                 v8:    (t=exp(t_prime) init 10, b init -10)
```

At the current v7 / v8 configuration (frozen text encoder, precomputed embeddings):

| Component | Params | Notes |
|---|---:|---|
| EfficientNet B1 backbone | 6.5 M | ImageNet-pretrained, trainable |
| Image projector | 1.6 M | 2-layer MLP (GELU): 1280 → 1024 → 512 |
| Text projector | 3.4 M | 2-layer MLP (GELU): 2560 → 1024 → 512 |
| Learned scalars | 1 (v6/v7) / 2 (v8) | temperature (+ bias in v8) |
| **Trainable total** | **11.5 M** | v6 was 118.6 M because the BGE text encoder trained too |

## Files

```
train_precomp.py      # main training entrypoint (precomp pipeline). Builds loader +
                      # prefetcher + transforms + model + optimizer, runs the
                      # eval-interval loop calling train_steps + validate.
train_utils.py        # train_steps inner loop, lr_schedule_factor, gather_features,
                      # clip_loss_gathered (v6/v7), siglip_loss (v8),
                      # save_ckpt, manage_ckpts.
eval_utils.py         # validate (1M-pair pass) + streaming_metrics (R@k + mean rank).
model.py              # CLIPPrecompModel (image bb + projectors + learned scalars) +
                      # Projector. CLIPModel (v6 dual-tower) kept for benchmark.py /
                      # evaluate.py compatibility.
dataset.py            # WebDataset streaming loader. JPEG decode + 256-crop → uint8 HWC
                      # + sample_idx (the cc12m stem parsed as int, used as memmap row).
benchmark.py          # v6 + v7 vs OpenAI CLIP / OpenCLIP cc12m on the 5000-pair set.
evaluate.py           # R@{1,5,10} on a single v6 checkpoint (legacy, used by v6 ckpts).
run_train.sh          # launcher: sets CUDA_DEVICE_ORDER=PCI_BUS_ID and the 5090 GPU
                      # mask, then accelerate launch train_precomp.py.
config.yaml           # all tunable hyperparameters.

scripts/              # reusable infrastructure (used by training and one-time jobs)
  embedding_extract.py    # 6-rank Qwen3-Embedding-4B extractor → text_emb.mmap
  run_extract.sh          # launcher for the 6× 5090 extraction job
  gpu_transforms.py       # GPU-side RRC + HFlip + ColorJitter + Normalize
  prefetcher.py           # PinnedPrefetcher (side stream + persistent pinned buffers)
```

## Setup

Tested on PyTorch 2.7 + CUDA 12.8, 6× RTX 5090.

```bash
pip install torch torchvision timm transformers accelerate webdataset braceexpand pyyaml pillow
```

Configure Accelerate once:

```bash
accelerate config
# pick: MULTI_GPU, num_processes = (how many GPUs you have), mixed_precision = bf16
```

Point the `TRAIN_SHARDS` / `VAL_SHARDS` lists at the top of `dataset.py` at your
cc12m tar files. They default to
`/mnt/md0/cc12m/cc12m-train-{0000..2175}.tar`, with shards 0020–0219 held out
as the validation range.

## Validation set

The 1M-pair val is streamed from cc12m shards **0020–0219** at eval time —
nothing to pre-build. Training reads the remaining 1976 shards
(0000–0019 + 0220–2175, ~9.96 M pairs). A small cached 5000-pair tensor
(`val_set_cc12m_0020_5000.pt`) is kept for `benchmark.py` — the
apples-to-apples comparison against OpenAI CLIP / OpenCLIP uses that
exact set.

## Train

```bash
VERSION=8 ./run_train.sh
```

The launcher (`run_train.sh`) sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=0,1,2,3,5,6` (the six 5090s; the 3090 at index 4 is
skipped), then hands off to `accelerate launch train_precomp.py`. Both the
env-hygiene and the `VERSION` env var are required. `VERSION=N` sets the
output dir to `runs/vN/`, which is wiped at startup — bump `N` per attempt.
Logs stream to stdout and `runs/vN/train.log`; the resolved config snapshot
is dumped as `runs/vN/config.snapshot.yaml`. Checkpoint management keeps
only `best` (highest avg held-out R@1) and `last` on disk.

## Evaluate

```bash
# 5000-pair apples-to-apples table: v6 + v7 vs OpenAI CLIP / OpenCLIP cc12m
CUDA_VISIBLE_DEVICES=0 python benchmark.py

# Single-checkpoint R@1/5/10 + mean rank (v6 CLIPModel only — pre-precomp models)
python evaluate.py runs/v6/checkpoints/ckpt_step00123196.pt
```

## Hyperparameters

Defaults in `config.yaml`:

| key | default | notes |
|---|---|---|
| `epochs` | 20 | ~40 min/epoch on 6×5090 at v7 settings; slower at v8's smaller image batch |
| `warmup_steps` | 2000 | linear → cosine to 0 |
| `base_lr` | 5e-4 | head LR; backbone LR = `base_lr × backbone_lr_mult` |
| `backbone_lr_mult` | 0.2 | gentler updates on the pretrained image backbone |
| `weight_decay` | 0.1 | excluded from biases, LayerNorm, learned scalars, ViT-specific tokens |
| `grad_clip` | 1.0 | |
| `eval_every_frac` | 1 | evals per epoch (~7 min per 1M-pair eval on 6×5090) |
| `log_every` | 50 | in-batch metrics cadence |
| `n_negs` | 1024 | **v8 only** — random text negatives per anchor per step, drawn from /dev/shm memmap ([#18](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/18) for why not larger yet) |

### Learned temperature / bias

- **v6/v7 (InfoNCE)**: single scalar `log_scale`, init 0 (scale 1), clamped at
  `exp(log_scale) ≤ 100`. Flat-softmax init forces real embedding separation
  before the temperature can artificially sharpen the loss.
- **v8 (SigLIP)**: two scalars `t = exp(t_prime)` with `t_prime` init `log(10)` (so
  `t = 10`) and `b` init `-10`. No clamp on `t` — the negative-shifted bias
  handles the heavy negative pool (paper §3.2).

## Notes on the setup

- The text encoder is frozen on purpose. Co-training a from-scratch text encoder
  alongside a from-scratch (or pretrained) image encoder on 10 M pairs is not a
  great use of capacity — frozen retrieval-trained text embeddings already give
  the text side a strong representation, and the image side learns to align to
  it. Original CLIP trained the text encoder from scratch, but they had 400 M
  pairs and many more GPUs.
- The MLP projectors use a hidden dim larger than the output dim (1024 → 512).
  Original CLIP uses a *linear* projection (no hidden layer); they argue
  non-linear projectors are co-adapted to self-supervised representation
  learning and don't help here. Worth A/B testing.

## Performance pipeline (v6+ speedups)

> **TL;DR.** Three stacked v6 bottlenecks fixed once — the frozen text encoder
> forward becomes a memmap lookup, CPU augmentation moves to the GPU, and the
> H2D copy of each image batch overlaps with the previous step's compute. Net
> effect: v7 runs at ~5,650 samples/s vs v6's ~2,800 — a **2× throughput win**
> on the same 6×5090 box.

| # | Bottleneck (before) | Approach | Win |
|---|---|---|---|
| [#8](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/8) | Frozen text encoder fwd ran every step (~50 ms/step, ~40% of compute) | Extract once with **Qwen3-Embedding-4B** + flash-attn-2 → fp16 **sparse memmap** keyed by `int(stem)` | text fwd → **0 ms/step** at train time; one-time ~25 min on 6×5090 |
| [#13](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/13) | albumentations CPU aug per sample → GPU stalled at 30-70% util | Move RandomResizedCrop + HFlip + ColorJitter + Normalize to GPU; RRC+flip fold into one affine matrix consumed by `F.grid_sample` | **~5.5 ms / B=256** on a 3090; CPU workers just decode + resize-to-256 |
| [#14](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/14) | 50 MB/batch sync H2D blocked next forward | `PinnedPrefetcher`: side stream + persistent pinned host buffers; H2D for batch N+1 overlaps with compute on batch N | **~11% wall-clock** on a synthetic 50-batch bench (4.65 s → 4.12 s) |

Worth-knowing gotcha caught while landing the above:
[#15](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/15) — `non_blocking=True` is silently sync without a pinned source. The classic prefetcher trap; a side-stream copy that *looks* async will silently fall back to a default-stream sync copy if the host tensor isn't pinned.

The training rewrite that actually consumes all three speedups is `train_precomp.py` — text tower removed, memmap row lookup at the index, `GPUTrainTransform`, `PinnedPrefetcher`.

## v6 vs v7 — what the pipeline actually bought us

> **TL;DR.** Pipeline wise, v7 wins outright: 2× throughput, 42 % less
> wall-clock, 270 ms saved per step. Quality wise, v7 *loses* by ~2 R@1
> on the same 1M-pair held-out val, because freezing the text encoder
> removed ~107 M of joint-adaptation capacity. Even a much stronger frozen
> encoder (Qwen3-Embedding-4B) doesn't make up for losing the ability to
> bend text toward the visual distribution. v7's role is as **the infra**,
> not as the model.

v6 and v7 ran the exact same 20-epoch schedule over the same train shards
on the same 6×5090 box. The only differences: v7's text tower is gone
(precomputed Qwen3-Embedding-4B embeddings in a /dev/shm memmap), images
are augmented on the GPU, and the prefetcher overlaps H2D with compute.

### Wall-clock

| | v6 | v7 | Δ |
|---|---|---|---|
| Throughput | ~2,800 samples/s | ~5,650 samples/s | **2.0×** |
| Total wall-clock (20 epochs) | 22.4 h | 12.9 h | **−9.5 h (−42%)** |
| Per-step time (B=1536 global) | ~540 ms | ~270 ms | **−270 ms / step** |

Where the 270 ms went, roughly:
- Text encoder forward (BGE base, B=256): **~50 ms** → 0 (memmap lookup)
- CPU augmentation (albumentations per sample, 8 workers): **~150 ms wait** → ~5 ms GPU (no wait)
- Sync H2D of (B, 256, 256, 3) uint8: **~10–15 ms** → hidden behind compute
- Misc (better worker overlap, less Python overhead on the worker side): **~60 ms**

### Held-out R@1

This is where the trade-off bites.

| | v6 | v7 |
|---|---|---|
| Architecture | Image bb + **trainable** BGE-base text | Image bb + **frozen** Qwen3-Embedding-4B (lookup) |
| Trainable params | 118.6 M | 11.5 M |
| 1M-pair val avg R@1 | **4.528 %** | **2.466 %** |
| 5000-pair val i2t R@1 | **45.12 %** | **34.12 %** |
| 5000-pair val t2i R@1 | **45.36 %** | **35.18 %** |

v7's pipeline runs ~2× faster but **its retrieval quality is worse**.
Reason: freezing the text encoder removes ~107 M of joint-adaptation
capacity. A much stronger frozen text encoder (Qwen3-Embedding-4B,
2560-dim, retrieval-tuned at LLM scale) doesn't make up for losing the
ability to bend the text representation toward the visual distribution.

The right framing of v7 isn't "v7 is better than v6." It's **"v7 is the
infra"** — we now have a scalable, throughput-friendly chassis. The
next step is to put *the right model* on top of it.

## Where we stand vs OpenAI CLIP & OpenCLIP — on the same 5000-pair eval

> **TL;DR.** Data quantity + quality are the biggest levers, not the recipe.
> OpenAI CLIP B/16 has similar trainable capacity to v6 (~120–150 M) but was
> trained on ~40× more, cleaner data (WIT-400M vs cc12m's 11 M) and clears
> v6 by +15 R@1. Even with infinite compute on cc12m we'd plateau in the
> mid-50s % because the captions are noisier and narrower than WIT.

| Model | Params | i2t R@1 | t2i R@1 | mean rank | Notes |
|---|---:|---:|---:|---:|---|
| **v7 (ours)** | 11.5 M | 34.12 % | 35.18 % | 36.5 | frozen text, cc12m, 20 epochs |
| **v6 (ours)** | 118.6 M | 45.12 % | 45.36 % | 32.4 | trainable text, cc12m, 20 epochs |
| OpenAI CLIP ViT-B/16 | 149.6 M | 60.66 % | 59.10 % | 23.5 | trainable text, **WIT-400M** |
| OpenAI CLIP ViT-L/14 | 427.6 M | 67.74 % | 66.82 % | 19.1 | trainable text, **WIT-400M**, larger image bb |
| OpenCLIP RN50 cc12m | 102.0 M | 84.58 % | 84.66 % | 1.6 | trainable text, **cc12m** (shard 0020 partially leaked) |

### What this says about scaling

- **More data is the biggest lever.** OpenAI CLIP B/16 has similar trainable
  capacity to v6 (~120–150 M) but was trained on ~40× more data (WIT-400M
  vs cc12m's 11 M) and clears v6 by **+15 R@1**. cc12m caption quality
  is much worse than WIT, so the gap is partly data scale and partly
  data quality.
- **Joint text training is also a big lever.** OpenCLIP RN50 on cc12m
  destroys both of us with **+39 R@1** vs v6 at fewer parameters. They
  trained the whole stack and ran for a long time. Their advantage is
  inflated some by shard-0020 leakage into their training set, but most
  of it is real.
- **What if we trained 5× longer on cc12m?** v7's loss + R@1 slope was
  still decreasing at epoch 19 (the cosine schedule was draining the
  LR, not the gradient signal). A longer schedule probably gets us
  another 1–2 R@1, but won't close the 15-point gap to CLIP B/16. That
  gap is the data.

### Would we get CLIP-quality results with comparable data?

Honest answer: **the recipe matters more than we hoped, and the data
matters more than the recipe.** With WIT-400M + a properly jointly-trained
text encoder + a ViT-class image bb, the v7-pipeline numbers would land
in CLIP B/16's neighborhood (~60 % R@1 on this 5000-pair eval). With
cc12m only — even with infinite compute — we plateau around
mid-50s % R@1 on the same eval, because cc12m captions just aren't as
clean or as broad as WIT.

The cleanest near-term direction is **SigLIP-style training**:
- Keep the text frozen (so v7's 2× pipeline speedup stays).
- Drop the image batch but pull *many* random text-embedding negatives
  per step (we already have them all in /dev/shm).
- Use sigmoid loss instead of softmax InfoNCE — it tolerates
  arbitrary numbers of negatives without needing a square K×K matrix.

This is what the `v8-siglip` branch implements. The next section reports
the numbers.

## v8 — SigLIP recovers the quality v7 lost ([branch](https://github.com/MokshitSama/clip-from-scratch-cc12m/tree/v8-siglip))

> **TL;DR.** Swap InfoNCE → sigmoid (SigLIP), drop image batch from 256 to 64,
> pull `n_negs=1024` random extra text negatives per step from the /dev/shm
> memmap. Each pair is independent BCE, so image and text batch sizes can
> differ freely — image forward is 4× cheaper, and the model still sees
> ~1088 candidates per anchor. Result at the epoch-9 checkpoint: **5.15 %
> R@1 on 1M-pair held-out val** — already above v6's *final* number (4.53 %)
> at less than half the schedule, on v7's pipeline. Still climbing.

v8 keeps v7's frozen-text pipeline but swaps two things:
- **Smaller image batch**, `BATCH_SIZE = 64` per rank (vs v7's 256). Image
  forward is the bottleneck, so 4× fewer images per step ≈ 4× cheaper.
- **Sigmoid loss** ([Zhai et al. 2023](https://arxiv.org/abs/2303.15343)),
  on a `(K=64) × (K+M=1088)` logit matrix where each step pulls
  `n_negs = 1024` random extra text embeddings from `/dev/shm` as
  negatives. Independent BCE per pair — no softmax denominator, no
  K=M constraint.

Two new learnable scalars replace the InfoNCE `log_scale`:
- `t` = `exp(t_prime)`, init at 10 (sharp sigmoid from step 0)
- `b`, init at -10 (bias prior toward "everything is a negative pair"
  so the huge negative pool doesn't dominate the first gradients —
  paper §3.2)

### One bug worth recording

First v8 run set `n_negs = 10_000` and was **17× slower than v7**
(325 sps vs 5650). The text-side data path ballooned by 40× and the 51 MB
sync H2D per step blocked all 6 ranks — see
[#18](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/18)
for the full root cause and verification. Dropping to `n_negs = 1024`
took throughput to **~3800 sps** with balanced GPU util across all 6
ranks. The `PinnedPrefetcher` only covered the image side; the text side
is still synchronous and will go through a similar prefetcher refactor
before raising `n_negs` again.

### Numbers so far (run still in progress)

Snapshot at step 233,442 / 518,760 (epoch 9 / 20, ~45 % through):

| | v6 | v7 | v8 (running, epoch 9) |
|---|---|---|---|
| Text encoder | trainable BGE-base | frozen Qwen3-Emb-4B (lookup) | frozen Qwen3-Emb-4B (lookup) |
| Loss | InfoNCE (gathered) | InfoNCE (gathered) | SigLIP (no gather) |
| Per-rank batch / negatives per anchor | 256 / 1535 | 256 / 1535 | **64 / 1088** |
| Trainable params | 118.6 M | 11.5 M | 11.5 M |
| Throughput | ~2,800 sps | ~5,650 sps | ~3,800 sps |
| 1M-pair val avg R@1 (best) | 4.528 % | 2.466 % | **5.15 %** *(epoch 9)* |
| vs v6 | — | −2.06 | **+0.62 (already)** |

v8 surpassed v6's *final* number at less than half its schedule. The
trajectory is still climbing (4.65 → 4.99 → 5.15 across the last three
epochs); the run is expected to land at 6.0–7.0 % by epoch 20.

The verdict the v7 writeup hoped for landed: **freezing the text encoder
costs you R@1 under InfoNCE, but SigLIP with many cheap negatives
recovers it.** The infra v7 built (precomputed embeddings, GPU aug,
async H2D for images) is what made v8 affordable on the same 6×5090 box.

## Run log

Every run lives in `runs/v{N}/` with a snapshot of its config and a `train.log`.
Held-out **avg R@1** = mean of `R@1 i2t` and `R@1 t2i` on the fixed validation set.
v1-v4 used a 5,000-pair val set (shard 0020). From v5 on, the val set is ~1M pairs
(shards 0020-0219), so the held-out numbers across that boundary aren't 1:1 comparable —
a 1M val is statistically more reliable but also a strictly harder problem.

| Version | Doing what? | Change vs prev | Prediction | Actual (held-out avg R@1) | vs prev best |
|---|---|---|---|---|---|
| v1 | ViT-B/16 from scratch, 5 epochs, lr 1e-3, log_scale init 2.659 | initial baseline | should train slowly | **failed** — scale pinned at clamp(100) by step 3k, loss flat at log(batch), killed early | — |
| v2 | ViT-B/16 from scratch, 5 epochs, lr 5e-4, log_scale init 0.0 | lower LR; flatter softmax init | scale climbs slowly, loss starts dropping post-warmup | scale healthy (3.99 at kill), loss 7.13, but held-out R@1 only 0.05% at 0.5 ep — killed early to swap backbone | no real progress |
| v3 | ResNet-50 from scratch, 5 epochs | swap backbone ViT → ResNet-50 (random init) | conv inductive bias learns faster than ViT in this data regime | **R@1 = 4.03%** at end of 5 epochs | +4.03 |
| v4 | ResNet-50 (pretrained ImageNet), 20 epochs | longer schedule; pretrained warm-start instead of random init | 20 epochs + pretrained should land ~8-15% | **R@1 = 6.70%** at epoch ~10.5 (run killed ~52% in, so this is a partial result) | +2.67 |
| v5 | EfficientNet B1 (pretrained) + BGE-base text encoder, 1M val, 20 epochs, eval 1x/epoch | replaced ResNet-50 → EfficientNet B1 (6.5M vs 23.5M); BERT-base → BGE-base-en-v1.5 (same params, retrieval-trained); val 5k → 1M; eval 4x → 1x per epoch | smaller image bb might cap representation; BGE should noticeably improve text side; 1M val is more reliable but harder than 5k → R@1 numbers may look *lower* even if model is better | _early run; superseded by v6_ | _n/a_ |
| v6 | EfficientNet B1 + **trainable** BGE-base text encoder, 20 epochs, eval 4x/epoch, 1M val | unfroze the text encoder so both backbones train jointly; otherwise same as v5 | trainable text should give text side flexibility to align to images → bigger R@1 lift than v5 | **R@1 = 4.528%** at epoch 19 (`ckpt_step00123196.pt`); 22.4 h on 6×5090 | best to date |
| v7 | EfficientNet B1 + **precomputed** Qwen3-Embedding-4B text (frozen), 20 epochs, eval 1x/epoch, 1M val | text encoder forward removed entirely (memmap row lookup); on-GPU augmentation; pinned-buffer async H2D; tested SigLIP-style infrastructure | with much stronger frozen text emb + same image bb, expect R@1 comparable or slightly better than v6; pipeline should be ~2× faster | **R@1 = 2.466%** (lower); 12.87 h on 6×5090 (1.74× faster wall-clock). Quality regressed because frozen text can't bend toward visual distribution; pipeline win is real | −2.06 vs v6 |
| v8 | Same frozen Qwen3-Emb-4B but **SigLIP loss** with per-rank batch 64 and `n_negs=1024` random text negatives per step; otherwise same as v7 | softmax InfoNCE → sigmoid BCE; image batch 256 → 64 (4× cheaper fwd); text negatives 1535 (in-batch) → 1088 (most pulled from /dev/shm); no DDP gather of features | sigmoid loss + many cheap negatives should recover the joint-adaptation R@1 v7 lost to freezing text, while staying on v7's pipeline | (run in progress) — **5.15 %** at step 233k of 519k (epoch 9 of 20). Already above v6's final. Trajectory still climbing; projecting 6–7 % by epoch 20. Early `n_negs=10_000` attempt was 17× slower than v7 — see [#18](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/18) | **+0.62** vs v6 (so far) |

## Notes on choosing each piece

- **Image backbone:** ResNet-50 was chosen over ViT-B/16 after from-scratch ViT
  failed to converge on cc12m within 5 epochs. EfficientNet B1 is the current
  choice — pretrained ImageNet weights, much smaller (~6.5 M trainable backbone),
  faster per step. Pretrained weights skip the "learn to see" phase entirely.
- **Text encoder:** Originally pretrained-frozen BERT-base. Switched to
  pretrained-frozen `BAAI/bge-base-en-v1.5` — same parameter count, but BGE was
  contrastively trained for retrieval, so its `[CLS]` representation is much
  more useful for matching captions to images than BERT's pooler output (which
  was tuned for sentence-pair classification).
- **Projector:** 2-layer MLP (1024 → 512). Original CLIP uses a *linear* projection
  and explicitly removes the SimCLR-style non-linear projector. Worth A/B-testing.
- **Validation:** 1M pairs (shards 0020-0219, streamed at eval time). Statistical
  noise floor on R@1 ≈ 0.01% — basically zero. 5k val was 0.07%.
- **Eval cadence:** 1× per epoch. At 1M pairs each eval takes ~10-20 min, so
  per-quarter-epoch evals would have doubled the wall-clock cost of every run.

## Status

Work in progress. Reproducing CLIP-quality retrieval numbers needs much more data than
cc12m alone (the original paper used 400 M pairs); this repo is more about understanding
the recipe end-to-end than about competing with the reference.
