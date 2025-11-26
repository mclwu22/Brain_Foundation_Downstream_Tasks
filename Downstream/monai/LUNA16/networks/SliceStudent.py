import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import your dinov3 backbone and LoRA
from networks.dinov3.dinov3.models.vision_transformer import vit_base
from networks.dinov3.dinov3.models.stage2.lora import LoRA


def add_lora_to_vit(model, r=4):
    import math
    for blk in model.blocks:
        old_qkv = blk.attn.qkv
        dim = old_qkv.in_features

        w_a_q = nn.Linear(dim, r, bias=False)
        w_b_q = nn.Linear(r, dim, bias=False)
        w_a_v = nn.Linear(dim, r, bias=False)
        w_b_v = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(w_a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(w_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b_q.weight)
        nn.init.zeros_(w_b_v.weight)

        lora_qkv = LoRA(old_qkv, w_a_q, w_b_q, w_a_v, w_b_v)
        lora_qkv.qkv.weight.data.copy_(old_qkv.weight.data)

        if old_qkv.bias is not None:
            lora_qkv.qkv.bias.data.copy_(old_qkv.bias.data)

        blk.attn.qkv = lora_qkv

    return model

# ====== Transformer encoder for slice-level modeling ======
class SliceTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: (B, N_slices, 768)
        return self.encoder(x)  # (B, N_slices, 768)


# ====== Attention Pooling ======
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.att = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, N, 768)
        w = self.att(x)  # (B, N, 1)
        w = torch.softmax(w, dim=1)  # attention weights
        return (w * x).sum(dim=1)  # (B, 768)


# ====== 4. universal classifier head (supports K classes) ======
class SliceStudent(nn.Module):
    """
    Downstream version (CE Loss, K-class classification)
    """
    def __init__(self, 
                 n_slices=32, 
                 lora_rank=4, 
                 ckpt_path=None,
                 embed_dim=768,
                 num_classes=3,
                 frozen = True):       # ★★ add universal K classes
        super().__init__()
        self.n_slices = n_slices

        # === build backbone ===
        self.student = vit_base(
            drop_path_rate=0.2,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True
        )

        # === load checkpoint (optional) ===
        if ckpt_path is not None:
            print(f"[SliceStudent] Loading checkpoint: {ckpt_path}")
            chkpt = torch.load(
                ckpt_path,
                map_location='cpu',
                weights_only=False
            )
            state_dict = chkpt['teacher']
            state_dict = {
                k.replace('backbone.', ''): v
                for k, v in state_dict.items()
                if 'ibot' not in k and 'dino_head' not in k
            }
            self.student.load_state_dict(state_dict, strict=False)

        # === slice-level Transformer ===
        self.slice_transformer = SliceTransformer(embed_dim=embed_dim, depth=2, num_heads=8)
        self.att_pool = AttentionPooling(embed_dim=embed_dim)

        # === universal classifier head ===
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)  # ★★ replace 1 → K classes
        )
        if frozen:
            for p in self.student.parameters():
                p.requires_grad = False

    def forward(self, volume):
        B, C, D, H, W = volume.shape

        # sample N slices
        idx = torch.linspace(0, D - 1, self.n_slices).long().to(volume.device)
        slices = volume[:, :, idx, :, :]       
        slices = slices.permute(0, 2, 1, 3, 4)
        slices = slices.reshape(B * self.n_slices, 1, H, W)
        slices = slices.repeat(1, 3, 1, 1)       # DINO expects 3ch

        # # 2D ViT features
        # out = self.student.forward_features(slices)
        # patch_tokens = out["x_norm_patchtokens"]  # (BN, 14*14, 768)

        # token_number = patch_tokens.shape[1]
        # patch_tokens = patch_tokens.reshape(B, self.n_slices, token_number, 768)
        # tokens = patch_tokens.reshape(B, self.n_slices * token_number, 768)

        # # transformer + pooling
        # tokens = self.slice_transformer(tokens)
        # vol_feat = self.att_pool(tokens)

        # # logits (B, K)
        # return self.cls_head(vol_feat)
        # ===== 2. ViT forward (per slice) =====
        out = self.student.forward_features(slices)

        # === (key change) grab CLS token only ===
        cls_tokens = out["x_norm_clstoken"]             # (B*N, 768)

        # reshape to (B, N, 768)
        cls_tokens = cls_tokens.reshape(B, self.n_slices, -1)

        # ===== 3. Mean pooling across slices =====
        volume_feat = cls_tokens.mean(dim=1)            # (B, 768)

        # ===== 4. classifier =====
        return self.cls_head(volume_feat)
