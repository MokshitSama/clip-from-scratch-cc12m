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

## Architecture

```
image (224x224) -> EfficientNet B1 (pretrained, trainable) -> 1280 -> MLP -> 512 ─┐
                                                                                    ├─ cosine sim
text  (≤64 tok) -> BGE-base-en-v1.5 (pretrained, frozen)    ->  768 -> MLP -> 512 ─┘
                                                                             ↑
                                                                 learned log-scale
                                                                 temperature (init 0)
```

| Component | Params | Notes |
|---|---:|---|
| EfficientNet B1 backbone | 6.5 M | ImageNet-pretrained, trainable |
| BGE-base-en-v1.5 backbone | 109 M | frozen, `[CLS]` of last hidden state |
| Image projector | 1.6 M | 2-layer MLP (GELU) |
| Text projector | 1.3 M | 2-layer MLP (GELU) |
| `log_scale` | 1 | learned |
| **Trainable total** | **9.7 M** | |

## Files

```
model.py              # CLIPModel + clip_loss + Projector
build_loader.py       # WebDataset streaming loader over cc12m shards
build_val_set.py      # one-shot script to materialize the held-out val tensors
train.py              # multi-GPU training loop with gathered-features contrastive loss
evaluate.py           # eval R@{1,5,10} + mean rank on a checkpoint
config.yaml           # all tunable hyperparameters

scripts/              # performance pipeline (precomputed embeddings + GPU aug + async H2D)
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

Point `build_loader.SHARDS` at your cc12m tar files (they default to
`/mnt/md0/cc12m/cc12m-train-{0000..2175}.tar` minus shard 0020, which is held out for
validation).

## Validation set

The val set is streamed from cc12m shards **0020-0219** at eval time (~1M pairs).
Nothing to pre-build — the loader just opens those shards each time eval runs.
Training reads shards 0000-0019 + 0220-2175 (~9.96 M pairs).

The earlier `build_val_set.py` script pre-extracted 5,000 tensors to disk — kept
around for historical compatibility but no longer used by `train.py`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 VERSION=3 accelerate launch train.py
```

`VERSION=N` sets the output directory to `runs/vN/`. `runs/vN/` is wiped at startup if it
exists, so just bump the version number for each new attempt. Logs stream to stdout and to
`runs/vN/train.log`; the resolved config snapshot is dumped alongside as
`config.snapshot.yaml`. Checkpoint management keeps only `best` (highest avg held-out R@1
so far) and `last` (most recent) on disk.

You can also override the version via the `--version` flag instead of `VERSION=...`:

```bash
accelerate launch train.py --version 3
```

## Evaluate

```bash
python evaluate.py runs/v3/checkpoints/ckpt_step00007137.pt
# or evaluate every checkpoint in a directory:
python evaluate.py runs/v3/checkpoints/  --csv runs/v3/eval.csv
```

## Hyperparameters

Defaults in `config.yaml`:

| key | default | notes |
|---|---|---|
| `epochs` | 5 | ~0.7 h/epoch on 6× 5090 |
| `warmup_steps` | 2000 | linear → cosine to 0 |
| `base_lr` | 5e-4 | bump to 1e-3 for ResNet from-scratch |
| `weight_decay` | 0.1 | excluded from biases, LayerNorm, `log_scale`, ViT-specific tokens |
| `grad_clip` | 1.0 | |
| `eval_every_frac` | 4 | 4 evals per epoch (~15 min on 6× 5090) |
| `log_every` | 50 | in-batch metrics cadence |

The contrastive temperature `log_scale` is a *learned* parameter (init 0 → scale 1), clamped
to `exp(log_scale) ≤ 100` to prevent runaway. Init at scale 1 (flat softmax) forces the model
to learn real embedding separation before the temperature can sharpen the loss artificially.

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

Profiling the v6 step revealed three stacked bottlenecks. Each one has its
own GitHub issue with measurements + fix details:

| # | Bottleneck (before) | Approach | Win |
|---|---|---|---|
| [#8](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/8) | Frozen text encoder fwd ran every step (~50 ms/step, ~40% of compute) | Extract once with **Qwen3-Embedding-4B** + flash-attn-2 → fp16 **sparse memmap** keyed by `int(stem)` | text fwd → **0 ms/step** at train time; one-time ~25 min on 6×5090 |
| [#13](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/13) | albumentations CPU aug per sample → GPU stalled at 30-70% util | Move RandomResizedCrop + HFlip + ColorJitter + Normalize to GPU; RRC+flip fold into one affine matrix consumed by `F.grid_sample` | **~5.5 ms / B=256** on a 3090; CPU workers just decode + resize-to-256 |
| [#14](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/14) | 50 MB/batch sync H2D blocked next forward | `PinnedPrefetcher`: side stream + persistent pinned host buffers; H2D for batch N+1 overlaps with compute on batch N | **~11% wall-clock** on a synthetic 50-batch bench (4.65 s → 4.12 s) |

Worth-knowing gotcha caught while landing the above:
[#15](https://github.com/MokshitSama/clip-from-scratch-cc12m/issues/15) — `non_blocking=True` is silently sync without a pinned source. The classic prefetcher trap; a side-stream copy that *looks* async will silently fall back to a default-stream sync copy if the host tensor isn't pinned.

The training rewrite that actually consumes all three speedups (`train_precomp.py` — text tower removed, memmap row lookup at the index, `GPUTrainTransform`, `PinnedPrefetcher`) is in progress.

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
| v5 | EfficientNet B1 (pretrained) + BGE-base text encoder, 1M val, 20 epochs, eval 1x/epoch | replaced ResNet-50 → EfficientNet B1 (6.5M vs 23.5M); BERT-base → BGE-base-en-v1.5 (same params, retrieval-trained); val 5k → 1M; eval 4x → 1x per epoch | smaller image bb might cap representation; BGE should noticeably improve text side; 1M val is more reliable but harder than 5k → R@1 numbers may look *lower* even if model is better | _to be filled in_ | _to be filled in_ |

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
