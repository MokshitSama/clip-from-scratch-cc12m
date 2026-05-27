import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class Projector(nn.Module):
    """2-layer MLP that maps backbone features into the shared embedding space."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPModel(nn.Module):
    """Pretrained image backbone + pretrained-trainable text encoder + projectors + learned temperature.

    Both backbones are pretrained and both trainable. The text encoder is
    initialised from a retrieval-trained model (BGE-base by default) rather
    than vanilla BERT — semantically much stronger starting point for the
    same parameter budget. Training jointly lets it adapt to the visual
    distribution.
    """
    def __init__(
        self,
        image_backbone="tf_efficientnet_b1.aa_in1k",
        text_backbone="BAAI/bge-base-en-v1.5",
        embed_dim=512,
        proj_hidden_dim=1024,
    ):
        super().__init__()

        self.image_backbone = timm.create_model(image_backbone, pretrained=True, num_classes=0)
        # add_pooling_layer=False skips BERT's pooler (Linear + Tanh on [CLS]).
        # We use last_hidden_state[:, 0] directly, so the pooler weights would
        # be unused — and DDP refuses to train models with unused parameters
        # ("Parameter indices which did not receive grad...").
        self.text_backbone = AutoModel.from_pretrained(text_backbone, add_pooling_layer=False)

        self.image_projector = Projector(self.image_backbone.num_features, proj_hidden_dim, embed_dim)
        self.text_projector  = Projector(self.text_backbone.config.hidden_size, proj_hidden_dim, embed_dim)

        # Learned temperature, parametrised in log-space. Init at 0 (scale=1) so
        # the model has to learn embedding separation before sharpening the softmax.
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images):
        feats = self.image_backbone(images)
        return self.image_projector(feats)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        # BGE-style: raw [CLS] hidden state (NOT pooler_output, which is BERT's
        # tanh-projected variant tuned for NSP, not retrieval).
        cls = out.last_hidden_state[:, 0]
        return self.text_projector(cls)

    def forward(self, images, input_ids, attention_mask):
        img_emb = F.normalize(self.encode_image(images), dim=-1)
        txt_emb = F.normalize(self.encode_text(input_ids, attention_mask), dim=-1)
        logit_scale = self.log_scale.exp().clamp(max=100.0)
        return img_emb, txt_emb, logit_scale


def clip_loss(img_emb, txt_emb, logit_scale):
    """Symmetric InfoNCE over a single batch (no DDP gather)."""
    logits = logit_scale * img_emb @ txt_emb.T
    labels = torch.arange(img_emb.shape[0], device=img_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


if __name__ == "__main__":
    # Quick shape + loss sanity check on dummy tensors.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel().to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:>13,}")
    print(f"Trainable params: {trainable:>13,}  ({100 * trainable / total:.1f}%)")

    B = 4
    images = torch.randn(B, 3, 224, 224, device=device)
    input_ids = torch.randint(0, 30000, (B, 64), device=device)
    attn_mask = torch.ones(B, 64, dtype=torch.long, device=device)

    img_emb, txt_emb, scale = model(images, input_ids, attn_mask)
    loss = clip_loss(img_emb, txt_emb, scale)

    print(f"img_emb {tuple(img_emb.shape)}  txt_emb {tuple(txt_emb.shape)}  scale {scale.item():.3f}")
    print(f"loss = {loss.item():.4f}  (expected near log({B}) = {torch.log(torch.tensor(float(B))).item():.4f})")
