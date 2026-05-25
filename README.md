# CLIP from scratch on cc12m

A small implementation of [CLIP](https://arxiv.org/abs/2103.00020) trained from scratch on
[CC12M](https://github.com/google-research-datasets/conceptual-12m) (~11M image–caption pairs).
The image encoder is a ResNet-50 trained from random init; the text encoder is a frozen
pretrained BERT-base used as a fixed feature extractor. Two small MLP projectors map both
into a shared 512-dim space, and a learned temperature scales the cosine-similarity logits.

Multi-GPU with 🤗 Accelerate, using the OpenCLIP "gather features" trick so that the
contrastive loss is computed over the full *global* batch rather than each rank's local
slice — i.e. on N GPUs at batch B per rank, every anchor sees N·B − 1 negatives, not B − 1.

## Architecture

```
image (224x224) -> ResNet-50 (random init, trainable)   -> 2048 -> MLP -> 512 ─┐
                                                                                 ├─ cosine sim
text  (≤64 tok) -> BERT-base (pretrained, frozen)        ->  768 -> MLP -> 512 ─┘
                                                                          ↑
                                                              learned log-scale
                                                              temperature (init 0)
```

| Component | Params | Notes |
|---|---:|---|
| ResNet-50 backbone | 23.5 M | from scratch |
| BERT-base backbone | 110 M | frozen |
| Image projector | 2.6 M | 2-layer MLP (GELU) |
| Text projector | 1.3 M | 2-layer MLP (GELU) |
| `log_scale` | 1 | learned |
| **Trainable total** | **27.4 M** | |

## Files

```
model.py          # CLIPModel + clip_loss + Projector
build_loader.py   # WebDataset streaming loader over cc12m shards
build_val_set.py  # one-shot script to materialize the held-out val tensors
train.py          # multi-GPU training loop with gathered-features contrastive loss
evaluate.py       # eval R@{1,5,10} + mean rank on a checkpoint
config.yaml       # all tunable hyperparameters
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

## Build the validation set (once)

```bash
python build_val_set.py
```

Materializes 5000 preprocessed (image, caption) pairs from cc12m shard 0020 into
`val_set_cc12m_0020_5000.pt` (~3 GB). Done once per dataset; reused across all runs.

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

## Notes on the from-scratch setup

- BERT is frozen on purpose. Training BERT from random init alongside a from-scratch ResNet on
  11M pairs would split a limited learning signal across too many parameters. A pretrained
  BERT gives the text side a usable representation immediately and lets the image side catch up.
- ResNet-50 was chosen over ViT-B/16 after a from-scratch ViT failed to converge in this
  data regime — the conv inductive bias makes ResNets learn useful image features far faster
  than ViTs on cc12m-sized datasets.
- The MLP projectors use a hidden dim larger than the output dim (1024 → 512). Without the
  hidden expansion, a single linear layer caps the achievable alignment between the two
  encoder outputs.

## Status

Work in progress. Reproducing CLIP-quality retrieval numbers needs much more data than
cc12m alone (the original paper used 400 M pairs); this repo is more about understanding
the recipe end-to-end than about competing with the reference.
