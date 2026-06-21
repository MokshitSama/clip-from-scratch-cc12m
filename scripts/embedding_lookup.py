# Using Accelerator for all gpu and putting data(memmap) into the System RAM(shm)

import json
import shutil
from pathlib import Path
import numpy as np

def load_embedding_table(src_dir: Path, shm_path: Path | None, accelerator):
    src_dir = Path(src_dir)
    meta = json.loads((src_dir/"meta.json").read_text())
    src_mmap = src_dir/meta["file"]

    if shm_path is None:
        target = src_mmap
    else:
        target = Path(shm_path)
        if accelerator.is_main_process and not target.exists():
            print(f"[shm] staging {src_mmap} -> {target} ...")
            shutil.copy(src_mmap, target)
            print(f"[shm] done ({target.stat().st_size / 1e9:.1f} GB)")
        accelerator.wait_for_everyone()

    mmap = np.memmap(
        target, dtype=np.dtype(meta["dtype"]),mode="r",
        shape = (meta["max_rows"], meta["dim"])
    )
    return mmap
