import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

class GPUTrainTransform(nn.Module):
    def __init__(self, image_size: int=IMAGE_SIZE,
                 scale=(0.7, 1.0),
                 ratio=(3/4, 4/3),
                 hflip_p = 0.5,
                 jitter_p = 0.5,
                 brightness = 0.1,
                 contrast = 0.1,
                 saturation = 0.05):
        super().__init__()
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_p = hflip_p
        self.jitter_p = jitter_p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def random_resized_crop(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device

        scale = torch.empty(B, device=device).uniform_(self.scale[0], self.scale[1])
        log_r = torch.empty(B, device=device).uniform_(
            float(torch.log(torch.tensor(self.ratio[0]))),
            float(torch.log(torch.tensor(self.ratio[1])))
        )

        ratio = log_r.exp()

        half_w = (scale * ratio).sqrt()
        half_h = (scale / ratio).sqrt()

        max_cx = (1.0 - half_w).clamp(min=0)
        max_cy = (1.0 - half_h).clamp(min=0)
        cx = torch.empty(B, device=device).uniform_(-1, 1) * max_cx
        cy = torch.empty(B, device=device).uniform_(-1, 1) * max_cy

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = half_w           # scale x
        theta[:, 1, 1] = half_h           # scale y
        theta[:, 0, 2] = cx               # translate x
        theta[:, 1, 2] = cy               # translate y

        grid = F.affine_grid(
            theta, [B, 3, self.image_size, self.image_size],
            align_corners=False,
        )

        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float in [0, 1]
        B = x.shape[0]
        device = x.device

        # Per-sample "apply" gate (1.0 = jitter this sample, 0.0 = identity).
        apply = (torch.rand(B, 1, 1, 1, device=device) < self.jitter_p).float()

        def factor(delta):
            # apply==0 -> 1.0; apply==1 -> 1.0 + U(-delta, +delta)
            return 1.0 + apply * (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * delta

        bright = factor(self.brightness)
        contr  = factor(self.contrast)
        satur  = factor(self.saturation)

        # 1. Brightness.
        x = x * bright

        # 2. Contrast around per-image mean.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * contr + mean

        # 3. Saturation: blend with per-pixel luma.
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]   # (B, 1, H, W)
        x = (x - luma) * satur + luma

        return x.clamp_(0.0, 1.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: uint8 (B, H, W, 3) on GPU
        x = x.permute(0, 3, 1, 2).contiguous().float()/255.0 # (b, 3, h, w) in [0, 1]
        x = self.random_resized_crop(x)
        x = torch.where(torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.hflip_p, x.flip(-1), x)
        x = self._color_jitter(x)
        x = (x - self.mean) / self.std
        return x


