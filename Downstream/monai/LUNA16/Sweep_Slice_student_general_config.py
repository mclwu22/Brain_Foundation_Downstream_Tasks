import sys
sys.path.append("/home/exx/Desktop/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16/networks/dinov3")

import os
import torch
from trainers.dinov3_volume2d_trainer import  dinov3_volume2d_trainer_ADNI

class slice_student_config:
    def __init__(self):

        # ---- Dataset ----
        self.dataset_name = "ADNI"  # ★ 只需要改这里

        # ---- General ----
        self.n_slices = 128
        self.lora_rank = 8
        self.ckpt_path = None
        self.attr = 'class'
        self.manualseed = 111
        self.model = "slice_student"
        # ---- Train ----
        self.batch_size = 2
        self.val_batch = 4
        self.lr = 1e-4
        self.epochs = 1
        self.val_epoch = 1
        self.patience = 10
        self.num_workers = 4
        self.train_ratio = 1.0
        self.gpu_ids = [0]
        self.scheduler = "ReduceLROnPlateau"
        # ---- logging ----
        self.save_root = "./slice_student_results"
        self.note = "slice_student"
        self.benchmark = False
    def display(self, logger=None):
        print("=== Slice Student Config ===")
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                print(f"{k:20s}: {getattr(self, k)}")
if __name__ == "__main__":
    config = slice_student_config()
    Trainer = dinov3_volume2d_trainer_ADNI(config)
    Trainer.train()
