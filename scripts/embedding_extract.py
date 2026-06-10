"""Embed every cc12m caption once, into one big lookup table.

Captions are never augmented, so a caption's embedding is identical every
epoch — there is no reason to recompute it during training. This script runs
the text model exactly once per caption and writes the vectors into a single
numpy memmap file. After that, training's whole text side is an array read.

The table is indexed by the sample's key: cc12m keys are globally unique
integers, so `table[int(key)]` is the lookup — no index structure to build or
keep in sync. Keys that never got a row (failed downloads, empty captions)
simply stay zero, and the loader drops pairs whose row is all-zero. The file
is created sparse, so unwritten rows take no actual disk space.

Multi-GPU without any coordination: every GPU process takes a stride of the
shard list (rank 0 gets shards 0, 6, 12...; rank 1 gets 1, 7, 13...) and
writes only its own samples' rows. The row ranges never overlap, so all six
processes write into the same file at full speed with no locks and no merge
step at the end. A `done/` marker per shard makes the whole thing resumable:
kill it anytime, rerun, it continues where it stopped.

Full run (GPUs 0-5 are the 5090s on this box; ~70 min for 11M captions):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/embedding_extract.py

Small test on the spare GPU, two shards into a throwaway dir:
    CUDA_VISIBLE_DEVICES=6 python scripts/embedding_extract.py --shards 0-1 \
        --out /mnt/md0/cc12m_qwen3emb_smoke
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp


MODEL_DEFAULT = "Qwen/Qwen3-Embedding-4B"
DIM           = 2560               # Qwen3-Embedding-4B output width
MAX_ROWS      = 12_500_000         # > cc12m's highest key; sparse file, unwritten rows cost nothing
DTYPE         = np.float16
SHARD_PATH    = "/mnt/md0/cc12m/cc12m-train-{:04d}.tar"


def parse_shards(spec):
    """'0-2175' or '0,1,2' -> list of shard numbers."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def read_shard(shard_idx):
    """All (key, caption) pairs of one shard. Empty captions are skipped —
    their rows stay zero and the training loader drops those pairs."""
    keys, caps = [], []
    path = SHARD_PATH.format(shard_idx)
    with tarfile.open(path) as tf:
        for m in tf:
            if m.name.endswith(".txt"):
                c = tf.extractfile(m).read().decode("utf-8", "replace").strip()
                if not c:
                    continue
                keys.append(int(m.name[:-4]))
                caps.append(c)
    return np.asarray(keys, dtype=np.int64), caps


def load_model(model_name, device):
    """sentence-transformers reads the correct pooling for the model from its
    own config — important, because decoder models like Qwen3 must be pooled
    at the LAST token (the only position that has seen the whole sentence),
    while BERT-style models pool at the first. Getting this wrong produces
    vectors that look fine and are garbage. FlashAttention-2 if available."""
    from sentence_transformers import SentenceTransformer
    for attn in ("flash_attention_2", "sdpa"):
        try:
            m = SentenceTransformer(
                model_name, device=device,
                model_kwargs={"attn_implementation": attn, "torch_dtype": torch.bfloat16},
            )
            print(f"[{device}] loaded {model_name} with attn={attn}", flush=True)
            return m
        except Exception as e:
            print(f"[{device}] attn={attn} failed ({str(e)[:60]}); trying next", flush=True)
    return SentenceTransformer(model_name, device=device,
                               model_kwargs={"torch_dtype": torch.bfloat16})


def worker(local_rank, args, world, mm_path, done_dir):
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    model = load_model(args.model, device)
    # Cap caption length so the longest batch has a known, bounded memory
    # cost. Without the cap, one batch of unusually long captions OOMed.
    model.max_seq_length = args.max_seq_len

    mm = np.memmap(mm_path, dtype=DTYPE, mode="r+", shape=(MAX_ROWS, DIM))

    all_shards = parse_shards(args.shards)
    my_shards = [s for s in all_shards[local_rank::world]
                 if not (done_dir / f"shard_{s:04d}").exists()]
    if not my_shards:
        print(f"[{device}] nothing to do (all shards done)", flush=True)
        return

    print(f"[{device}] {len(my_shards)} shards to embed", flush=True)
    t0 = time.time()
    n_done = 0

    # Read the next shard's captions on a thread while the GPU embeds the
    # current one — tar reading and embedding overlap instead of alternating.
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(read_shard, my_shards[0])
    for i, shard in enumerate(my_shards):
        try:
            keys, caps = fut.result()
        except Exception as e:
            print(f"[{device}] read shard {shard} failed: {str(e)[:80]}", flush=True)
            if i + 1 < len(my_shards):
                fut = ex.submit(read_shard, my_shards[i + 1])
            continue
        if i + 1 < len(my_shards):
            fut = ex.submit(read_shard, my_shards[i + 1])

        if len(caps) == 0:
            (done_dir / f"shard_{shard:04d}").write_text("0\n")
            continue

        emb = model.encode(
            caps, batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        mm[keys] = emb.astype(DTYPE)               # scatter into this shard's rows
        (done_dir / f"shard_{shard:04d}").write_text(f"{len(caps)}\n")

        n_done += 1
        if n_done % 5 == 0 or i == len(my_shards) - 1:
            seen = sum(int((done_dir / f"shard_{s:04d}").read_text() or 0)
                       for s in my_shards[:i + 1] if (done_dir / f"shard_{s:04d}").exists())
            rate = seen / (time.time() - t0 + 1e-9)
            print(f"[{device}] {n_done}/{len(my_shards)} shards | "
                  f"{seen:,} caps | {rate:.0f} caps/s", flush=True)

    mm.flush()
    print(f"[{device}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/md0/cc12m_qwen3emb")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--shards", default="0-2175", help="'0-2175' or '0,1,2'")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=128,
                    help="truncate captions to this many tokens (bounds peak memory)")
    args = ap.parse_args()

    out = Path(args.out)
    done_dir = out / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    mm_path = str(out / "text_emb_qwen3-4b_fp16.mmap")

    # Allocate the full (sparse) file up front so workers can open it 'r+'.
    if not Path(mm_path).exists():
        np.memmap(mm_path, dtype=DTYPE, mode="w+", shape=(MAX_ROWS, DIM)).flush()

    # meta.json is how the training side knows the shape/dtype to open with.
    with open(out / "meta.json", "w") as f:
        json.dump({"model": args.model, "dim": DIM, "max_rows": MAX_ROWS,
                   "dtype": "float16", "normalized": True, "row_index": "int(key)",
                   "file": Path(mm_path).name}, f, indent=2)

    world = torch.cuda.device_count()
    if world == 0:
        raise RuntimeError("no CUDA devices visible")
    print(f"launching {world} GPU worker(s) | out={out} | shards={args.shards}", flush=True)

    if world == 1:
        worker(0, args, 1, mm_path, done_dir)
    else:
        mp.spawn(worker, args=(args, world, mm_path, done_dir), nprocs=world, join=True)

    n_shards_done = len(list(done_dir.glob("shard_*")))
    n_caps = sum(int((p.read_text() or 0)) for p in done_dir.glob("shard_*"))
    print(f"\nALL DONE: {n_shards_done} shards, {n_caps:,} captions embedded -> {mm_path}")


if __name__ == "__main__":
    main()
