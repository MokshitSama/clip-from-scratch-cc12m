"""GPU-side image augmentation and the pinned-memory prefetcher.

Both pieces came out of profiling the training step (the [prof] lines in the
train log break each step into phases — that's how the costs below were
found).

Augmentation: we first used kornia on the GPU; it measured ~990ms per
512-image batch — 82% of the entire step. The same augmentations as plain
tensor ops measure ~12ms. Two things make the native version fast: the random
crop and the random flip collapse into a single warp call over the whole
batch, and color jitter with hue=0 (our config) never leaves RGB, so it is
just a few multiply-adds. Hue is the only expensive jitter component — it
needs a colorspace round trip — and we don't use it.

Transfer: copying a batch CPU->GPU is only fast from "pinned" memory (pages
locked in place, so the GPU can DMA directly from them; from ordinary memory
the copy goes through a hidden staging step at ~1GB/s). PyTorch's built-in
pin_memory pins fresh memory for every batch, all run long, and that crashed
on this machine mid-training ("CUDA error: invalid argument"). PinnedPrefetcher
pins a handful of buffers once at startup and reuses them forever; a small
background thread keeps them filled so the next batch is already staged when
the training loop asks. Step cost of getting data onto the GPU: ~355ms before,
~15ms after.
"""
import math
import queue
import threading
import time

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _rrc_flip(x, out_size, scale, ratio=(3.0 / 4.0, 4.0 / 3.0), p_flip=0.5):
    """RandomResizedCrop + RandomHorizontalFlip for the whole batch in one
    grid_sample call.

    Per image we draw: kept area (`scale`), crop aspect (`ratio`), position
    (tx, ty), and a flip coin. Each draw becomes a tiny 2x3 matrix ("theta")
    describing which rectangle of the source to read and at what orientation;
    grid_sample then resamples all images through their own matrices at once.
    The flip costs nothing extra — mirroring is the x-axis times -1 inside a
    warp that is happening anyway.
    """
    B = x.shape[0]
    dev = x.device
    a = torch.empty(B, device=dev).uniform_(*scale)
    r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    wf = (a * r).sqrt().clamp(max=1.0)              # crop width as a fraction of the image
    hf = (a / r).sqrt().clamp(max=1.0)              # crop height as a fraction of the image
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - wf)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - hf)
    flip = torch.where(torch.rand(B, device=dev) < p_flip, -1.0, 1.0)

    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = wf * flip
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = hf
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _color_jitter(x, brightness, contrast, saturation, p):
    """Brightness / contrast / saturation with independent random strength
    per image. `p` gates jitter per image; an image that sits out gets a
    factor of exactly 1.0, i.e. passes through unchanged.

    brightness scales every value; contrast pulls values toward or away from
    the image's mean; saturation blends toward or away from the grayscale
    version of the image.
    """
    B = x.shape[0]
    dev = x.device
    apply = (torch.rand(B, 1, 1, 1, device=dev) < p).float()

    def factor(amount):
        f = torch.empty(B, 1, 1, 1, device=dev).uniform_(1.0 - amount, 1.0 + amount)
        return 1.0 + (f - 1.0) * apply

    x = x * factor(brightness)
    gray = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor(contrast) + mean
    x = gray + (x - gray) * factor(saturation)
    return x.clamp_(0.0, 1.0)


class GPUTransform:
    """Batched augmentation + normalization on the GPU.

    Input is the loader's raw uint8 [B, H, W, 3] tensor. Train: random crop
    to 224 + flip + jitter + normalize. Val: images already arrive at 224,
    so only scale to [0,1] and normalize — validation must be deterministic.

    All randomness is drawn per image, not per batch: two copies of the same
    photo in one batch get different crops. (Some libraries draw one set of
    random values for the whole batch, which quietly weakens augmentation.)
    """
    def __init__(self, device, image_size=224, scale=(0.7, 1.0),
                 brightness=0.1, contrast=0.1, saturation=0.05,
                 p_jitter=0.5, p_flip=0.5):
        self.image_size = image_size
        self.scale = scale
        self.brightness, self.contrast, self.saturation = brightness, contrast, saturation
        self.p_jitter, self.p_flip = p_jitter, p_flip
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    def _to_float(self, x):
        # bytes (0..255, channels last) -> floats (0..1, channels first),
        # done on the GPU so the small uint8 form is what crossed the bus.
        return x.permute(0, 3, 1, 2).float().div_(255.0)

    @torch.no_grad()
    def train(self, x):
        x = self._to_float(x)
        x = _rrc_flip(x, self.image_size, self.scale, p_flip=self.p_flip)
        x = _color_jitter(x, self.brightness, self.contrast, self.saturation, self.p_jitter)
        return x.sub_(self.mean).div_(self.std)

    @torch.no_grad()
    def val(self, x):
        return self._to_float(x).sub_(self.mean).div_(self.std)


class PinnedPrefetcher:
    """Keeps the next batch staged in pinned memory so the GPU never waits
    for data.

    At construction, `n_slots` buffers are allocated in pinned memory — once,
    on the main thread, at a quiet moment. (Pinning repeatedly mid-training
    is what crashed: same failure class the repo hit before in e5badff.)

    A daemon thread (the feeder) pulls batches from the loader and memcpys
    them into free slots — pure CPU work. The training loop calls .next():
    take a filled slot, start an asynchronous copy to the GPU (fast, since
    the source is pinned), hand the slot back for refilling. One CUDA event
    per slot stops the feeder from overwriting a buffer whose GPU copy hasn't
    finished — without it, a fast feeder could corrupt a batch in flight.

    The feeder restarts the loader when it runs out, so .next() never raises
    StopIteration; the training loop just counts its own steps.
    """
    def __init__(self, loader, device, batch_spec, n_slots=4, log=print):
        self.loader = loader
        self.device = device
        self.filled = queue.Queue()
        self.free   = queue.Queue()
        self.slots  = []
        for _ in range(n_slots):
            bufs = [torch.empty(shape, dtype=dtype, device="cpu") for shape, dtype in batch_spec]
            try:
                bufs = [b.pin_memory() for b in bufs]
            except RuntimeError as e:
                log(f"[prefetcher] WARNING: pin_memory failed ({str(e)[:60]}) — "
                    f"falling back to UNPINNED staging (H2D will be slower)")
            slot = {"bufs": bufs, "ev": torch.cuda.Event(), "ev_recorded": False}
            self.slots.append(slot)
            self.free.put(slot)
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._started = False
        self._wait_s = 0.0          # time .next() spent waiting on an empty queue

    def pop_wait_ms(self):
        """Starved time since last asked, in ms. ~0 means the pipeline keeps
        up; growing values mean the GPU is outrunning the data."""
        w, self._wait_s = self._wait_s, 0.0
        return w * 1000.0

    def _feed(self):
        # "Current GPU" is per-thread in torch; a fresh thread defaults to
        # GPU 0, which is wrong for every rank except rank 0. Bind first.
        torch.cuda.set_device(self.device)
        while True:
            for batch in self.loader:
                slot = self.free.get()
                if slot["ev_recorded"]:
                    slot["ev"].synchronize()       # the slot's previous GPU copy must finish first
                for buf, t in zip(slot["bufs"], batch):
                    buf.copy_(t)                   # CPU-to-CPU memcpy into pinned memory
                self.filled.put(slot)

    def next(self):
        """Returns the next batch as GPU tensors. Call from the main thread."""
        if not self._started:
            self._thread.start()
            self._started = True
        t0 = time.perf_counter()
        slot = self.filled.get()
        self._wait_s += time.perf_counter() - t0
        out = tuple(buf.to(self.device, non_blocking=True) for buf in slot["bufs"])
        slot["ev"].record()                        # slot becomes reusable once the copy passes this point
        slot["ev_recorded"] = True
        self.free.put(slot)
        return out
