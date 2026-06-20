import numpy as np
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from sentence_transformers import SentenceTransformer
import torch
import tarfile
import time
import json

from pathlib import Path
import glob
ALL_SHARDS = sorted(glob.glob("/mnt/md0/cc12m/cc12m-train-*.tar"))

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MAX_ROWS = 12_500_500
DIM = 2560
MMAP_NAME = "text_emb.mmap"  

def iter_shard_captions(shard_path):
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.name.endswith(".txt"):
                continue
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                text = f.read().decode("utf-8").strip()
            except Exception:
                continue
            if not text:
                continue
            stem = member.name[:-4] #striping ".txt"
            yield stem, text

def extract_shard(model, mmap, shard_path, done_dir, batch_size=256):
    shard_num = int(Path(shard_path).stem.split("-")[-1])
    marker = done_dir/f"shard_{shard_num:04d}"
    if marker.exists():
        print(f" skip shard shard_{shard_num:04d} (already done)")
        return
    keys, texts = [], []

    for stem, text in iter_shard_captions(shard_path):
        keys.append(int(stem))
        texts.append(text)

    if not texts:
        marker.write_text("0\n")
        return
    
    t0 = time.time()
    embs = model.encode(
        texts, batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float16)

    keys_arr = np.asarray(keys, dtype=np.int64)
    mmap[keys_arr] = embs

    dt = time.time() - t0
    print(f"  shard {shard_num:04d}: {len(texts)} captions in {dt:.1f}s "
          f"({len(texts)/dt:.0f} caps/sec)")
    
    marker.write_text(f"{len(texts)}\n")



def load_model(device="cuda:0"):
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = SentenceTransformer(MODEL_NAME, device=device, 
                                        model_kwargs={"torch_dtype":torch.bfloat16,
                                        "attn_implementation":attn}
                                        )
            print(f"loaded {MODEL_NAME} with attn {attn}")
            return model
        except Exception as e:
            print(f"attn={attn} failed ({str(e)[:80]})")
    raise RuntimeError("alll attention implementation failed")



#----------------MAIN---------------------


OUTPUT_DIR = Path("/mnt/md0/cc12m_qwen3_emb_4b_embeddings")
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    my_shards = ALL_SHARDS[rank::world_size]
    print(f"[Rank {rank}/{world_size}] {len(my_shards)} shards to process")


    print(f"loading {MODEL_NAME}...")
    model = load_model("cuda:0")
    model.max_seq_length = 128


    mmap_path = OUTPUT_DIR / MMAP_NAME
    done_dir = OUTPUT_DIR / "done"
    done_dir.mkdir(parents=True, exist_ok=True)

    if not mmap_path.exists():
        print(f"allocating sparse memmap at {mmap_path}(shape ({MAX_ROWS}, {DIM}))...")
        np.memmap(mmap_path, dtype=np.float16, mode="w+",
                  shape=(MAX_ROWS, DIM)).flush()
        print(" done")

    mmap = np.memmap(mmap_path, dtype=np.float16, mode="r+", 
                     shape=(MAX_ROWS, DIM))

    for i, shard in enumerate(my_shards):
        print(f"\n[Rank {rank}] shard {i+1}/{len(my_shards)}: {Path(shard).name}")
        extract_shard(model, mmap, shard, done_dir, batch_size=128)
        #break


    with open(OUTPUT_DIR/"meta.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "dim" : DIM, "max_rows": MAX_ROWS,
            "dtype": "float16", "normalized": True,
            "row_index": "int(cc12m_stem)",
            "file": MMAP_NAME,
        }, f, indent=2)

if __name__ == "__main__":
    main()