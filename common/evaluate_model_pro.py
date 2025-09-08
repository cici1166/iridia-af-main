# eval_simple.py  — simple, English-only evaluator (Val/Test Accuracy, F1, AUC)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, sys
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- ensure project root is on sys.path so "common.model_v2" can be imported ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------------------------

from common.model_v2 import AFPredictionModelV2  # your model class

# ----------------- config (edit if your paths differ) -----------------
H5_PATH   = os.path.join("dataset", "processed_data.h5")
VAL_CSV   = os.path.join("splits", "val.csv")
TEST_CSV  = os.path.join("splits", "test.csv")
CKPT_CANDIDATES = ["best_model_v2.pt", "best_model.pt"]  # picked in order
BATCH_SIZE = 16
# ---------------------------------------------------------------------

class ECGDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        self.h5_path = h5_path
        self.samples = pd.read_csv(csv_path)
        self.h5_file = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        row = self.samples.iloc[idx]
        ecg = self.h5_file[row["sample_id"]][:]  # (120000,) or (120000, 2)
        if ecg.ndim == 2:
            ecg = ecg[:, 0]                     # use first channel if 2D
        x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # (1, L)
        y = 1 if row["segment_label"] == "pre_af" else 0
        return x, y

def forward_probs(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze()
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(np.atleast_1d(prob))
            y_true.extend(y)
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)

def eval_set(name, model, loader, device):
    y, p = forward_probs(model, loader, device)
    # when only one class present in y, AUC is undefined
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, zero_division=0)
    return {"Set": name, "Accuracy": acc, "F1": f1, "AUC": auc}

def main():
    print("=== Eval (simple) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # pick checkpoint
    ckpt_path = None
    for c in CKPT_CANDIDATES:
        if os.path.exists(c):
            ckpt_path = c
            break
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found: best_model_v2.pt / best_model.pt")

    # data
    val_ds = ECGDataset(H5_PATH, VAL_CSV)
    test_ds = ECGDataset(H5_PATH, TEST_CSV)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = AFPredictionModelV2().to(device)
    ckpt = torch.load(ckpt_path, map_location=device，weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path}")

    # evaluate
    val_row  = eval_set("Val",  model, val_loader, device)
    test_row = eval_set("Test", model, test_loader, device)

    # pretty table
    df = pd.DataFrame([val_row, test_row])
    df_round = df.copy()
    for c in ["Accuracy", "F1", "AUC"]:
        df_round[c] = df_round[c].astype(float).round(4)

    print("\n== Summary (thr=0.5) ==")
    print(df_round.to_string(index=False))

    # save csv
    os.makedirs("eval_outputs", exist_ok=True)
    out_csv = os.path.join("eval_outputs", "metrics_simple.csv")
    df_round.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("=== Done ===")

if __name__ == "__main__":
    main()
