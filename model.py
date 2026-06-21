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
    
class CLIPPrecompModel(nn.Module):
    """Image backbone + image projector + text projector + learned scale.

    Text tower is gone — text embeddings come from a precomputed memmap
    (Qwen3-Embedding-4B, 2560-dim, already L2-normalized) keyed by
    sample_idx. The text_projector maps that fp16 2560-dim vector into
    the shared embed_dim (512 by default).
    """
    def __init__(self,
                 image_backbone="tf_efficientnet_b1.aa_in1k",
                 text_emb_dim=2560,
                 embed_dim=512,
                 proj_hidden_dim=1024):
        super().__init__()
        self.image_backbone  = timm.create_model(image_backbone, pretrained=True, num_classes=0)
        self.image_projector = Projector(self.image_backbone.num_features, proj_hidden_dim, embed_dim)
        self.text_projector  = Projector(text_emb_dim, proj_hidden_dim, embed_dim)
        self.log_scale       = nn.Parameter(torch.tensor(0.0))

    def encode_image(self, images):
        return self.image_projector(self.image_backbone(images))
    
    def encode_text(self, text_emb):
        return self.text_projector(text_emb.float())
    
    def forward(self, images, text_emb):
        img = F.normalize(self.encode_image(images), dim=-1)
        txt = F.normalize(self.encode_text(text_emb), dim=-1)
        return img, txt, self.log_scale.exp().clamp(max=100.0)




# Not be using but keeping for the sake of previous versions

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


if __name__ == "__main__":
    # Smoke test for CLIPPrecompModel: dummy images + a dummy 2560-dim text emb,
    # verify shapes through the projectors + that the loss math wires up.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPPrecompModel().to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CLIPPrecompModel — Total params:     {total:>13,}")
    print(f"CLIPPrecompModel — Trainable params: {trainable:>13,}  "
          f"({100 * trainable / total:.1f}%)")

    B = 4
    images   = torch.randn(B, 3, 224, 224, device=device)
    text_emb = F.normalize(torch.randn(B, 2560, device=device), dim=-1)   # mimic memmap row

    img_emb, txt_emb, scale = model(images, text_emb)
    print(f"img_emb {tuple(img_emb.shape)}  txt_emb {tuple(txt_emb.shape)}  "
          f"scale {scale.item():.3f}")

    # Simple symmetric InfoNCE (positives on the diagonal) — random embs should
    # land near log(B) since chance accuracy is 1/B.
    logits = scale * img_emb @ txt_emb.T
    labels = torch.arange(B, device=device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    print(f"loss = {loss.item():.4f}  "
          f"(random anchors should land near log({B}) = {torch.log(torch.tensor(float(B))).item():.4f})")