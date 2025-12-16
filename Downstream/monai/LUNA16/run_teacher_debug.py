from Sweep_Slice_student_general_config import slice_student_config
from trainers.dinov3_volume2d_trainer_general import dinov3_volume2d_trainer
import traceback
import os

TEACHERS = ["BrainMVP", "bm_mae", "dinov3", "meddinov3"]
DATASETS = ["ABIDE", "ADNI", "BraTS", "UPENN"]
RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Global skip log (one line per skipped run)
GLOBAL_SKIP_FILE = "./SWEEP_RESULT/skip.txt"
os.makedirs(os.path.dirname(GLOBAL_SKIP_FILE), exist_ok=True)

for t in TEACHERS:
    for ds in DATASETS:
        for r in RATIOS:

            cfg = slice_student_config()
            cfg.dataset_name = ds
            cfg.train_ratio = r
            cfg.epochs = 1000 if t == "meddinov3" else 200
            cfg.teacher = True if t in ["bm_mae"] else False
            cfg.encoder_name = t
            cfg.note = f"{ds}_{t}_ratio_{r}"
            cfg.save_root = f"./SWEEP_RESULT/{cfg.note}"
            cfg.lr = 1e-5 if t.lower() == "scratch" else 1e-4

            print(f"\n🚀 Running teacher debug: {cfg.note}")

            try:
                trainer = dinov3_volume2d_trainer(cfg)
                trainer.train()

            except Exception as e:
                print(f"\n❌ ERROR in run: {cfg.note}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {e}")
                traceback.print_exc()

                # Append one-line record to global skip.txt
                with open(GLOBAL_SKIP_FILE, "a") as f:
                    f.write(f"{cfg.note}\n")

                print("➡️ Skipped and recorded in global skip.txt\n")
                continue
