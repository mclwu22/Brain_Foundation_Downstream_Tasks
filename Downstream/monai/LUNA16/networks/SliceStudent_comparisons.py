import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import your dinov3 backbone and LoRA
from networks.dinov3.dinov3.models.vision_transformer import vit_base
from networks.dinov3.dinov3.models.stage2.lora import LoRA
import os
import sys

def load_teacher_unified(teacher_name, dune_base="/scr2/yz_wu/foundation/dune/models/mf", device="cuda"):
    """
    Unified teacher loader for Stage-2 pretraining.

    Args:
        teacher_name (str): one of {"bm_mae", "brainmvp", "sam_med3d", "brainiac"}.
        dune_base (str): path to the DUNE teacher module folder (contains builder.py, config.py).
        device (str): device to load the model on.

    Returns:
        torch.nn.Module: teacher model (already .eval() and .cuda()).
    """


    print("🔍 [DEBUG] Starting to load teacher:", teacher_name)
    print(f"🔍 [DEBUG] DUNE base path: {dune_base}")
    print(f"🔍 [DEBUG] Device: {device}")

    try:
        builder_path = os.path.join(dune_base)
        if builder_path not in sys.path:
            sys.path.append(builder_path)
            print(f"📁 [DEBUG] Added {builder_path} to sys.path")

        # 导入 builder 模块
        print("🧩 [DEBUG] Importing build_teachers from dinov3.models.stage2.builder ...")
        from dinov3.models.stage2.builder import build_teachers

        print(f"🧠 [DEBUG] Building teacher '{teacher_name}' ...")
        teachers = build_teachers([teacher_name])
        print(f"✅ [DEBUG] build_teachers() returned: {list(teachers.keys())}")

        teacher = teachers[teacher_name]

        print("🚀 [DEBUG] Moving teacher to device ...")
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        print(f"✅ [SUCCESS] Teacher '{teacher_name}' loaded and moved to {device}")

        return teacher

    except Exception as e:
        print(f"❌ [ERROR] Failed to load teacher '{teacher_name}': {e}")

        raise


class SliceStudent_comparisons(nn.Module):
    """
    Downstream version (CE Loss, K-class classification)
    """
    def __init__(self, 
                 n_slices=32, 
                 lora_rank=4, 
                 ckpt_path=None,
                 num_classes=3,
                 frozen = True,
                 teacher_name="bm_mae",
                 ):       # ★★ add universal K classes
        super().__init__()
        self.n_slices = n_slices
        self.teacher_name = teacher_name
        dune_base = "/home/exx/Desktop/Med_DINOv3/Transfer/mf"  # your DUNE path
        self.teacher = load_teacher_unified(self.teacher_name , dune_base)
        if self.teacher_name == "brainmvp":
            self.embed_dim = 512     # BrainMVP final feature map has 512 channels
        else:
            self.embed_dim = 768     # BM-MAE, DINO, ViT etc.

        # === universal classifier head ===
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, num_classes)  # ★★ replace 1 → K classes
        )
        if frozen:
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, volume):
        B, C, D, H, W = volume.shape
        with torch.no_grad():

            t = self.teacher(volume)                                  # [B, 513,768]
            if self.teacher_name  =="brainmvp":
                # t is a 5-element tuple of multi-level feature maps
                final_feat = t[-1]                 # [B, 512, 14, 8, 14]
                t_cls = final_feat.mean(dim=[2,3,4])  # [B, 512]
            else:
                t_cls = t[:, 0, :]  # CLS token

        # ===== 4. classifier =====
        return self.cls_head(t_cls)
