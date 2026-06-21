import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

AMP_DTYPE = torch.bfloat16


def lr_schedule_factor(step, total_steps, warmup):
    """Returns a multiplier in [0, 1]: linear warmup, then cosine decay to 0.
    Applied per-param-group against that group's base_lr in the train loop."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def gather_features(local_img, local_txt, accelerator):
    """all_gather embeddings across ranks, keeping local-slice gradients."""
    if accelerator.num_processes == 1:
        return local_img, local_txt

    ws   = accelerator.num_processes
    rank = accelerator.process_index

    img_buf = [torch.zeros_like(local_img) for _ in range(ws)]
    txt_buf = [torch.zeros_like(local_txt) for _ in range(ws)]
    dist.all_gather(img_buf, local_img)
    dist.all_gather(txt_buf, local_txt)
    # all_gather is non-differentiable; splice the local tensor back in so
    # gradients flow into the local model.
    img_buf[rank] = local_img
    txt_buf[rank] = local_txt
    return torch.cat(img_buf, dim=0), torch.cat(txt_buf, dim=0)


def clip_loss_gathered(local_img, local_txt, logit_scale, accelerator):
    """Symmetric InfoNCE over the global batch.

    Returns (loss, R@1_i2t, R@1_t2i) where the R@1s are measured on local
    anchors against global candidates (so the printed in-batch number is
    against world_size * batch candidates, not just the local batch).
    """
    all_img, all_txt = gather_features(local_img, local_txt, accelerator)
    rank = accelerator.process_index
    bs   = local_img.shape[0]

    logits_i2t = logit_scale * local_img @ all_txt.T
    logits_t2i = logit_scale * local_txt @ all_img.T
    labels = torch.arange(bs, device=local_img.device) + rank * bs

    loss = 0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))

    with torch.no_grad():
        acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean()
        acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean()
    return loss, acc_i2t, acc_t2i


def save_ckpt(path, step, model, optimizer):
    """Save the full model + optimizer state. Both backbones are trainable
    now, so we can't shortcut by skipping the text encoder anymore.
    Checkpoint size jumps from ~50MB to ~500MB+ because of this."""
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def manage_ckpts(new_path, new_score, best_path, best_score, last_path):
    """Keep only the best-so-far and the most-recent checkpoint on disk."""
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


def train_steps(*, n_steps, model, prefetcher_iter, prefetcher,
                gpu_transform, text_table, optimizer, accelerator, cfg,
                start_step, total_steps, log, log_state):
    device = accelerator.device
    global_batch = accelerator.num_processes * cfg["per_rank_batch"]   # see note
    step = start_step
    end_step = start_step + n_steps
    while step < end_step:
        try:
            batch = next(prefetcher_iter)
        except StopIteration:
            prefetcher_iter = iter(prefetcher)
            batch = next(prefetcher_iter)

        imgs_uint8, idx = batch                             # both already on GPU (prefetcher)
        images = gpu_transform(imgs_uint8)                   # (B,3,224,224) float

        # CHANGE 1: text comes from memmap, not from a forward pass
        text_emb_np = text_table[idx.cpu().numpy()]          # (B, 2560) fp16 numpy
        text_emb = torch.from_numpy(text_emb_np).to(device, non_blocking=True)


        # Same cosine schedule for both groups, just scaled by each group's base_lr.
        factor = lr_schedule_factor(step, total_steps, cfg["warmup_steps"])
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * factor

        with autocast(device_type="cuda", dtype=AMP_DTYPE):
            # CHANGE 2: model signature is (images, text_emb), not (images, ids, mask)
            img_emb, txt_emb, logit_scale = model(images, text_emb)
            loss, acc_i2t, acc_t2i = clip_loss_gathered(
                img_emb, txt_emb, logit_scale, accelerator)

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"],
        )
        optimizer.step()

        with torch.no_grad():
            accelerator.unwrap_model(model).log_scale.clamp_(max=math.log(100.0))

        # Running stats for log_every (CHANGE 3: state lives in log_state dict,
        # not local vars, so it survives across train_steps calls)
        log_state["loss"] += loss.item()
        log_state["i2t"]  += acc_i2t.item()
        log_state["t2i"]  += acc_t2i.item()
        log_state["n"]    += 1

        if (step + 1) % cfg["log_every"] == 0:
            n = log_state["n"]
            now = time.time()
            sps = (cfg["log_every"] * global_batch) / (now - log_state["t_log"])
            head_lr = optimizer.param_groups[2]["lr"]
            bb_lr   = optimizer.param_groups[0]["lr"]
            log(f"step {step+1:6d}/{total_steps} | "
                f"loss {log_state['loss']/n:.4f} | scale {logit_scale.item():.2f} | "
                f"lr {head_lr:.2e}/{bb_lr:.2e} (head/bb) | "
                f"R@1 i2t {log_state['i2t']/n*100:5.1f}% | R@1 t2i {log_state['t2i']/n*100:5.1f}% | "
                f"{sps:.0f} samples/s")
            log_state["loss"] = log_state["i2t"] = log_state["t2i"] = 0.0
            log_state["n"] = 0
            log_state["t_log"] = now

        step += 1

    return step, prefetcher_iter
