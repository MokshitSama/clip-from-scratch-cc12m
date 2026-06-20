import numpy as np
import torch

class PinnedPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._iter = None
        self._next = None

        self._host_bufs: dict = {}

    def _get_pinned(self, shape, dtype):
        key = (tuple(shape), dtype)
        buf = self._host_bufs.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._host_bufs[key] = buf
        return buf

    def _to_gpu(self, batch):
        out = []
        for item in batch:
            if isinstance(item, np.ndarray):
                src = torch.from_numpy(item)
            else:
                src = item

            buf = self._get_pinned(src.shape, src.dtype) # persistent pinned tensor
            buf.copy_(src) # CPU memcpy: numpy → pinned

            out.append(buf.to(self.device, non_blocking=True))
        return tuple(out)
    
    def _preload(self):
        try:
            cpu_batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            self._next = self._to_gpu(cpu_batch)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self
    
    def __next__(self):
        if self._next is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        self._preload()
        return batch


