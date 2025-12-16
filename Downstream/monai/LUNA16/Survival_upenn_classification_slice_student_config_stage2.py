import sys
sys.path.append("/home/exx/Desktop/Med_DINOv3/Finetune/Large-Scale-Medical/Downstream/monai/LUNA16/networks/dinov3")

import os
import torch
from trainers.dinov3_volume2d_trainer import  dinov3_volume2d_trainer_Survival_upenn
from trainers.dinov3_stage2_volume2d_trainer import  dinov3_stage2_volume2d_trainer_Survival_upenn_proj,dinov3_stage2_volume2d_trainer_Survival_upenn_mlp_cls,dinov3_stage2_volume2d_trainer_Survival_upenn_mean_pooling,dinov3_stage2_volume2d_trainer_Survival_upenn_attn_pooling
class slice_student_config:
    def __init__(self):

        # ---- Dataset ----
        self.data_root = "/home/exx/Desktop/Med_DINOv3/Datasets/Finetune_classification/Survival"
        self.train_csv = "train.csv"
        self.val_csv = "val.csv"
        self.train_dataset = "survival_upenn"
        self.model = "slice_student"
        self.network = "slice_student_3d"
        self.note = f"slice_student_{self.train_dataset}_attn_pooling_test"
        self.attr = 'class'
        # ---- Model ----
        self.network = "slice_student"  # ★ must match networks_dict
        self.n_slices = 128              # how many slices to sample
        self.lora_rank = 8              # LoRA rank
        self.ckpt_path = "/home/exx/Desktop/Med_DINOv3/Transfer/DINOV3-Stage2-rope_cls_token_vitb_lora_epoch60_slice_128_batch_4_train_attn_mlp/stage2_cls_token_epoch10.pth"  # your pretrained student
        self.manualseed = 111  # 666
        # ---- Training ----
        self.gpu_ids = [0]
        self.batch_size = 16
        self.val_batch = 4
        self.benchmark = False
        self.lr = 1e-4
        self.optim = "adam"
        self.loss = "bce"
        self.scheduler = "ReduceLROnPlateau"
        self.epochs = 100
        self.val_epoch = 1
        self.num_workers = 4
        self.patience = 10
        self.frozen = True  # freeze student backbone
        # ---- Logging ----
        self.verbose = 1
        self.max_queue_size = self.num_workers * 1

        # ---- Others ----
        self.save_root = "./slice_student_results"

    def display(self, logger=None):
        print("=== Slice Student Config ===")
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                print(f"{k:20s}: {getattr(self, k)}")



if __name__ == "__main__":
    config = slice_student_config()
    # Trainer = dinov3_volume2d_trainer_Survival_upenn(config)
    # Trainer = dinov3_stage2_volume2d_trainer_Survival_upenn_mean_pooling(config)
    Trainer = dinov3_stage2_volume2d_trainer_Survival_upenn_attn_pooling(config)
    Trainer.train()
