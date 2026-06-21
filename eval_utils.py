import torch
from torch.amp import autocast

AMP_DTYPE = torch.bfloat16
SIM_CHUNK = 512

@torch.no_grad()
def streaming_metrics(img_emb, txt_emb, chunk=SIM_CHUNK):
    """R@{1,5,10} + mean_rank in both directions, computed in query-chunks.

    Avoids materialising the full NxN similarity matrix (which would be 4 TB
    at N=1M). Memory peaks at chunk x N x 4 bytes (2 GB at chunk=512, N=1M).

    Releases the CUDA caching allocator's free blocks before allocating the
    first big sims chunk — training fragments the cache, and even though
    plenty of memory is technically free, the contiguous block we need can
    fail to allocate without empty_cache (see issue #6).
    """
    torch.cuda.empty_cache()
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
def validate(*, model, val_loader, gpu_val_transform, text_table,
             device, log, max_pairs=None, progress_every_batches=50):
    """One pass over val_loader, accumulating embeddings on GPU,
    then chunked retrieval metrics.

    model:             unwrapped CLIPPrecompModel (call inside `if is_main:`,
                       pass `accelerator.unwrap_model(model)`)
    val_loader:        yields (uint8_imgs HWC, sample_idx int64) per batch
    gpu_val_transform: GPUValTransform (deterministic resize + normalize)
    text_table:        np.memmap[(max_rows, dim)] of precomputed text emb
    device:            accelerator.device
    log:               your logger callable
    max_pairs:         optional cap (e.g. 1_000_000) on # pairs to embed

    Returns (metrics_dict, n_pairs_seen).
    """
    model.eval()
    img_chunks, txt_chunks = [], []
    n_seen = 0
    n_batches = 0

    with autocast(device_type="cuda", dtype=AMP_DTYPE):
        for imgs_uint8, idx in val_loader:
            imgs_uint8 = imgs_uint8.to(device, non_blocking=True)
            imgs = gpu_val_transform(imgs_uint8)

            # text comes from the memmap, keyed by sample_idx
            text_emb_np = text_table[idx.numpy()]               # (B, 2560) fp16
            text_emb    = torch.from_numpy(text_emb_np).to(device, non_blocking=True)

            img_emb, txt_emb, _ = model(imgs, text_emb)
            img_chunks.append(img_emb.float())
            txt_chunks.append(txt_emb.float())

            n_seen   += imgs.shape[0]
            n_batches += 1
            if n_batches % progress_every_batches == 0:
                log(f"    [eval] embedded {n_seen:,} pairs...")

            if max_pairs is not None and n_seen >= max_pairs:
                break

    model.train()

    img_emb = torch.cat(img_chunks, dim=0)
    txt_emb = torch.cat(txt_chunks, dim=0)
    if max_pairs is not None and img_emb.shape[0] > max_pairs:
        img_emb = img_emb[:max_pairs]
        txt_emb = txt_emb[:max_pairs]

    # Free the per-batch chunk lists, then release cached blocks so
    # streaming_metrics' big sims allocations have contiguous room.
    del img_chunks, txt_chunks
    torch.cuda.empty_cache()

    log(f"    [eval] embedded {img_emb.shape[0]:,} pairs total; "
        f"computing retrieval metrics...")
    metrics = streaming_metrics(img_emb, txt_emb)
    n_pairs = img_emb.shape[0]

    del img_emb, txt_emb
    torch.cuda.empty_cache()
    return metrics, n_pairs
