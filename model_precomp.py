"""Model for the precomputed-text pipeline.

The original setup ran two networks every step: the image encoder and a 109M-
parameter text encoder. But captions never change — there is no augmentation
on text, so a caption produces the same embedding every single epoch.
Recomputing it 20 times is wasted work. Instead, every caption was embedded
once, offline, by a larger frozen text model (Qwen3-Embedding-4B), and saved
into one lookup table (scripts/embedding_extract.py). At training time:

    image   -> EfficientNet (trainable) -> image projector -> 512-dim vector
    caption -> table lookup             -> text projector  -> 512-dim vector

The projectors are small MLPs whose job is to bring both sides into the same
512-dim space, where an image and its caption should point in the same
direction. Trainable parameters dropped from ~120M to ~11.5M, and the text
tower's forward+backward cost dropped to zero.

log_scale is a single learned number that sharpens the softmax in the loss.
It starts at 0 (scale 1, a flat softmax) so the model has to learn real
separation before it can get confident, and it is clamped so it can't run
away — early runs with a hot start saw it pin at the clamp while the loss
went flat.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Projector(nn.Module):
    """2-layer MLP mapping a backbone's features into the shared space."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPPrecomp(nn.Module):
    def __init__(
        self,
        image_backbone="tf_efficientnet_b1.aa_in1k",
        text_emb_dim=2560,            # output width of Qwen3-Embedding-4B
        embed_dim=512,
        proj_hidden_dim=1024,
    ):
        super().__init__()
        self.image_backbone = timm.create_model(image_backbone, pretrained=True, num_classes=0)
        self.image_projector = Projector(self.image_backbone.num_features, proj_hidden_dim, embed_dim)
        self.text_projector  = Projector(text_emb_dim, proj_hidden_dim, embed_dim)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images):
        return self.image_projector(self.image_backbone(images))

    def encode_text(self, text_emb):
        # text_emb is the precomputed, already-normalized Qwen3 vector. This
        # MLP is the only trainable text compute in the pipeline.
        return self.text_projector(text_emb)

    def forward(self, images, text_emb):
        img_emb = F.normalize(self.encode_image(images), dim=-1)
        txt_emb = F.normalize(self.encode_text(text_emb), dim=-1)
        logit_scale = self.log_scale.exp().clamp(max=100.0)
        return img_emb, txt_emb, logit_scale


def clip_loss(img_emb, txt_emb, logit_scale):
    """Symmetric InfoNCE over one batch. (The multi-GPU version that sees all
    ranks' pairs is in train_precomp.py.)

    For every image, its own caption should out-score the other B-1 captions
    in the batch, and the same in the caption->image direction. That is a
    classification problem where pair i's correct class is i — hence
    cross_entropy against labels 0..B-1, the diagonal of the score matrix.
    """
    logits = logit_scale * img_emb @ txt_emb.T
    labels = torch.arange(img_emb.shape[0], device=img_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


if __name__ == "__main__":
    # Wiring check. At random init every caption is an equally plausible
    # match, so the loss must come out near log(B) — "1 out of B" confusion.
    # Anything else means shapes, normalization or labels are wrong.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPPrecomp().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    B, D = 4, 2560
    images = torch.randn(B, 3, 224, 224, device=device)
    text_emb = F.normalize(torch.randn(B, D, device=device), dim=-1)
    img_emb, txt_emb, scale = model(images, text_emb)
    loss = clip_loss(img_emb, txt_emb, scale)
    print(f"img_emb {tuple(img_emb.shape)} txt_emb {tuple(txt_emb.shape)} scale {scale.item():.3f}")
    print(f"loss {loss.item():.4f}  (expect ~log({B})={torch.log(torch.tensor(float(B))).item():.4f})")
