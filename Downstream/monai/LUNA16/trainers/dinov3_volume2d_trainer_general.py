from trainers.base_trainer import BaseTrainer
from sklearn import metrics
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
import torch.nn as nn

# ★ add your dataloader
from datasets_3D.Classification.abide_classification import ABIDEClassificationSet
from datasets_3D.Classification.adni_classification import ADNIClassificationSet
from datasets_3D.Classification.brats_2023_classification import BraTS_2023_ClassificationSet
from datasets_3D.Classification.Survival_upenn_classification import Survival_upenn_ClassificationSet
from networks.SliceStudent import SliceStudent,SliceStudent_attn_pooling
from networks.SliceStudent_comparisons import SliceStudent_comparisons

DATASET_REGISTRY = {
    "ABIDE": {
        "dataset": ABIDEClassificationSet,
        "num_classes": 2,
        "data_root": "/home/exx/Desktop/Med_DINOv3/Datasets/Finetune_classification/ABIDE/ABIDE_Classification/",
        "csv_format": "two_csv"  # train_0.csv + train_1.csv
    },

    "ADNI": {
        "dataset": ADNIClassificationSet,
        "num_classes": 3,
        "data_root": "/home/exx/Desktop/Med_DINOv3/Datasets/Finetune_classification/ADNI/",
        "csv_format": "single_csv"
    },

    "BraTS": {
        "dataset": BraTS_2023_ClassificationSet,
        "num_classes": 4,
        "data_root": "/home/exx/Desktop/Med_DINOv3/Datasets/Finetune_classification/Modality/BraTS_2023/",
        "csv_format": "single_csv"
    },

    "UPENN": {
        "dataset": Survival_upenn_ClassificationSet,
        "num_classes": 2,
        "data_root": "/home/exx/Desktop/Med_DINOv3/Datasets/Finetune_classification/Survival/",
        "csv_format": "single_csv"
    }
}
BACKBONE_REGISTRY = {
    "meddinov3": {
        "feat_dim": 768,
        "ckpt": "/home/exx/Desktop/Med_DINOv3/Models/Meddinov3/high_res/57999/model.pth",
    },
    "dinov3": {
        "feat_dim": 768,
        "ckpt": "/home/exx/Desktop/Med_DINOv3/pretrained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    },
    "BrainMVP": {
        "feat_dim": 768,
        "ckpt": "/home/exx/Desktop/Med_DINOv3/Transfer/mf/BrainMVP_uniformer.pt",
    },
    "bm_mae": {
        "feat_dim": 768,
        "ckpt": "/home/exx/Desktop/Med_DINOv3/Transfer/mf/bmmae.pth",
    },
    "scratch": {
        "feat_dim": 768,
        "ckpt": None,      # ★ 无 checkpoint
    }
}


class dinov3_volume2d_trainer(BaseTrainer):
    """
    A trainer for 2.5D models:
    3D volume → slice sampler → 2D encoder → aggregation → classifier.
    Compatible with BaseTrainer.
    """

    def __init__(self, config):
        super().__init__(config, init_dataloader=False)  # ★ 禁用 BaseTrainer dataloader

        # -----------------------------------------
        # 🔥 Build dataloaders here
        # -----------------------------------------
        info = DATASET_REGISTRY[config.dataset_name]

        DatasetClass = info["dataset"]
        data_root    = info["data_root"]
        num_classes  = info["num_classes"]

        backbone_info = BACKBONE_REGISTRY[config.encoder_name]
        print(f"[Trainer] Using backbone: {config.encoder_name}")
        self.ckpt_path = backbone_info["ckpt"]
        teacher = config.teacher

        self.train_dataset = DatasetClass(
            config,
            data_root,
            flag="train",
            train_ratio=config.train_ratio,
            teacher=teacher
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.eval_dataset = DatasetClass(
            config,
            data_root,
            flag="valid",
            teacher=teacher
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.start_epoch = 0

        TEACHERS = ["brainmvp", "bm_mae"]
        enc = config.encoder_name.lower()

        if enc in TEACHERS:
            self.model = SliceStudent_comparisons(
                ckpt_path=self.ckpt_path,
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=num_classes,
                teacher_name=enc
            )
        else:
            self.model = SliceStudent(
                ckpt_path=self.ckpt_path, # if None: scratch, if not, we use different path for dinov3 and meddinov3
                n_slices=config.n_slices,
                lora_rank=config.lora_rank,
                num_classes=num_classes,
                backbone_name=enc,
            )
        # 2. GPU
        self.model_to_gpu()

        # 3. loss
        self.criterion = nn.CrossEntropyLoss()

        # 4. optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=1e-5
        )

        # 5. scheduler
        if config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None
        print(f"[Trainer] Loaded {len(self.train_dataloader.dataset)} train volumes")
        print(f"[Trainer] Loaded {len(self.eval_dataloader.dataset)} val volumes")

    # ---------------------------
    # 🔥 override: set input
    # ---------------------------
    def set_input(self, sample):
        """
        sample: (volume, label, index)
        volume shape: (B,1,Nslices,H,W)
        """
        self.volume = sample[0].to(self.device)
        self.target = sample[1].to(self.device)

    # ---------------------------
    # 🔥 override: forward
    # ---------------------------
    def forward(self):
        logits = self.model(self.volume)          # (B,2)
        self.target = self.target.long().view(-1) # CE 需要 long
        self.loss = self.criterion(logits, self.target)
        self.pred = logits

    def backward(self):
        self.loss.backward()
    # ---------------------------
    # 🔥 main training loop
    # ---------------------------
    def train(self):
        train_losses = []
        avg_train_losses = []
        avg_valid_metrics = []

        best_metric = 0
        num_epoch_no_improvement = 0
        sys.stdout.flush()

        for epoch in range(self.start_epoch, self.config.epochs):
            # reset
            train_losses = []
            
            self.model.train()
            self.recorder.logger.info(
                'Epoch: %d/%d lr %e',
                epoch, self.config.epochs,
                self.optimizer.param_groups[0]['lr']
            )

            train_bar = tqdm(self.train_dataloader)
            for itr, sample in enumerate(train_bar):
                self.set_input(sample)
                self.optimize_parameters()
                train_losses.append(self.loss.item())

            self.recorder.writer.add_scalar('Train/total_loss', np.mean(train_losses), epoch)

            # ---------------------------
            # 🔥 Validation
            # ---------------------------
            if epoch % self.config.val_epoch == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.recorder.logger.info("validating....")

                    gts = []
                    preds = []

                    for itr, (vol, gt, _) in enumerate(self.eval_dataloader):
                        vol = vol.to(self.device)
                        gt = gt.to(self.device)

                        pred = self.model(vol)                    # (B, num_classes)
                        prob = torch.softmax(pred, dim=1)         # (B, num_classes)

                        gts.extend(gt.cpu().numpy())
                        preds.extend(prob.cpu().numpy())

                    gts = np.asarray(gts)         # (N,)
                    preds = np.asarray(preds)     # (N, C)

                    # ======================================================
                    # 🔥 Universal AUC for binary & multi-class
                    # ======================================================
                    # build one-hot ground truth: (N, C)
                    gts_onehot = np.zeros_like(preds)
                    gts_onehot[np.arange(len(gts)), gts] = 1

                    auc = 100.0 * metrics.roc_auc_score(
                        gts_onehot,
                        preds,
                        multi_class="ovr",
                        average="macro"
                    )
                    valid_metric = auc

                    self.recorder.writer.add_scalar('Val/metric', valid_metric, epoch)

                # logging
                train_loss = np.mean(train_losses)
                avg_train_losses.append(train_loss)
                avg_valid_metrics.append(valid_metric)

                self.recorder.logger.info(
                    "Epoch {}, validation AUC is {:.4f}, training loss is {:.4f}".format(
                        epoch + 1, valid_metric, train_loss
                    )
                )

                # Save results
                data_frame = pd.DataFrame(
                    data={
                        'Train_Loss': avg_train_losses,
                        'Val_AUC': avg_valid_metrics,
                    }
                )
                data_frame.to_csv(
                    os.path.join(self.recorder.save_dir, "results.csv"),
                    index_label='epoch'
                )

                # Early stopping
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    num_epoch_no_improvement = 0
                    self.save_state_dict(epoch + 1, os.path.join(self.recorder.save_dir, "model_best.pth"))
                else:
                    num_epoch_no_improvement += 1

                if num_epoch_no_improvement == self.config.patience:
                    self.recorder.logger.info("Early Stopping")
                    break

                if self.scheduler is not None:
                    self.scheduler.step(valid_metric)



        self.recorder.logger_shutdown()
        self.recorder.writer.close()
        return
