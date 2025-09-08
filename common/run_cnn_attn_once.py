# common/run_cnn_attn_once.py
# 一键训练 + 评估 CNN+Attention；把日志写入文件并在 Windows 上自动打开结果。

import os, csv, random, sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import h5py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------- 配置（按需改） ----------------
H5_PATH      = "dataset/processed_data.h5"
TRAIN_CSV    = "splits/train.csv"
VAL_CSV      = "splits/val.csv"
TEST_CSV     = "splits/test.csv"

CKPT_PATH    = "best_model_cnn_attn.pt"
RESULTS_CSV  = "results_val_test.csv"
LOG_PATH     = os.path.join("run_logs", "cnn_attn_log.txt")

SEED         = 42
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 0.0

# 参数量微调（与主模型±10%对齐时可改）
WIDTH_MULT   = 1.0
FC_DIM       = 128
ATTN_DIM     = 128  # 目前未用于 MHA 的 embed_dim，占位

# ---------------- 简易日志工具 ----------------
def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d): os.makedirs(d, exist_ok=True)

_ensure_dir(LOG_PATH)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def open_if_windows(path):
    if sys.platform.startswith("win") and os.path.exists(path):
        try:
            os.startfile(os.path.abspath(path))  # 打开 Notepad/默认程序
        except Exception as e:
            log(f"无法自动打开 {path}: {e}")

# ---------------- 工具函数 ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        prob = torch.sigmoid(model(xb))
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    if not ys:
        return {"accuracy": float("nan"), "f1": float("nan"), "auc": float("nan")}
    y = np.concatenate(ys); p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.int64)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}

def append_csv(path, row):
    _ensure_dir(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","split","accuracy","f1","auc"])
        if write_header: w.writeheader()
        w.writerow(row)

# ---------------- 数据集 ----------------
class ECGH5Dataset(Dataset):
    """
    兼容两种 CSV：
      A) group,index,label
      B) path,label   （path 形如 'pre_af/1234' 或 'nsr/5678'）
    不做逐段标准化（与主模型一致）
    """
    def __init__(self, h5_path: str, csv_path: str, normalize: bool = False):
        self.h5_path = h5_path
        self.csv = pd.read_csv(csv_path)
        self.normalize = normalize

        cols = {c.lower() for c in self.csv.columns}
        if {"group","index","label"} <= cols:
            self.mode = "group_index"
            self.csv.rename(columns={"group":"group","index":"index","label":"label"}, inplace=True)
        elif {"path","label"} <= cols:
            self.mode = "path"
            self.csv.rename(columns={"path":"path","label":"label"}, inplace=True)
        else:
            raise ValueError(f"CSV 列名不匹配：需要 (group,index,label) 或 (path,label) —— {csv_path}")

        self._h5 = None

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self): return len(self.csv)

    def _get_signal(self, row):
        h5 = self._open_h5()
        if self.mode == "group_index":
            grp = row["group"]; idx = int(row["index"])
        else:
            grp, idx = str(row["path"]).split("/")
            idx = int(idx)
        sig = np.array(h5[grp][idx], dtype=np.float32)  # (T,)
        return sig, int(row["label"])

    def __getitem__(self, i):
        row = self.csv.iloc[i]
        sig, label = self._get_signal(row)
        if self.normalize:
            std = sig.std() + 1e-6
            sig = (sig - sig.mean()) / std
        x = torch.from_numpy(sig).unsqueeze(0)  # (1, T)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def __del__(self):
        try:
            if self._h5 is not None: self._h5.close()
        except Exception:
            pass

# ---------------- 模型：CNN + MultiheadAttention ----------------
class CNNWithAttention(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels=(32, 64, 128, 128),
        kernel_sizes=(7, 5, 5, 3),
        strides=(4, 2, 2, 2),
        fc_dim: int = FC_DIM,
        attn_dim: int = ATTN_DIM,   # 预留
        dropout: float = 0.5,
        width_mult: float = WIDTH_MULT,
    ):
        super().__init__()
        chs = [max(1, int(c * width_mult)) for c in conv_channels]

        layers = []
        prev = in_channels
        for c, k, s in zip(chs, kernel_sizes, strides):
            layers += [
                nn.Conv1d(prev, c, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            prev = c
        self.feature = nn.Sequential(*layers)   # (B, C, T')

        # 动态选择 num_heads，尽量用 8，否则退到 4/2/1，避免“通道数不能整除头数”的报错
        heads = next((h for h in (8,4,2,1) if prev % h == 0), 1)
        self.attention = nn.MultiheadAttention(embed_dim=prev, num_heads=heads, dropout=dropout, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(prev, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x):            # x: (B, 1, T)
        x = self.feature(x)          # (B, C, T')
        x = x.transpose(1, 2)        # (B, T', C)
        attn_out, _ = self.attention(x, x, x)  # (B, T', C)
        x = attn_out.mean(dim=1)     # (B, C)
        logits = self.classifier(x)  # (B, 1)
        return logits.squeeze(-1)

# ---------------- 训练 + 评估一次性执行 ----------------
class EarlyStopper:
    def __init__(self, patience=8):
        self.patience = patience
        self.best = None
        self.bad = 0
    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric; self.bad = 0; return False
        self.bad += 1
        return self.bad > self.patience

def main():
    # 清空日志
    _ensure_dir(LOG_PATH)
    with open(LOG_PATH, "w", encoding="utf-8") as f: f.write("")

    log(f"file: {os.path.abspath(__file__)}")
    log(f"cwd : {os.getcwd()}")
    log(f"py  : {sys.executable}")

    # 基础路径检查
    missing = [p for p in [H5_PATH, TRAIN_CSV, VAL_CSV, TEST_CSV] if not os.path.isfile(p)]
    if missing:
        for m in missing:
            log(f"[ERROR] not found: {os.path.abspath(m)}")
        open_if_windows(LOG_PATH)
        return

    set_seed(SEED)
    pin = torch.cuda.is_available()

    # 数据
    train_ds = ECGH5Dataset(H5_PATH, TRAIN_CSV, normalize=False)
    val_ds   = ECGH5Dataset(H5_PATH, VAL_CSV,   normalize=False)
    test_ds  = ECGH5Dataset(H5_PATH, TEST_CSV,  normalize=False)
    log(f"#train={len(train_ds)}  #val={len(val_ds)}  #test={len(test_ds)}")

    # 训练数据均衡（与主模型一致）
    labels = [int(y.item()) for _, y in train_ds]
    class_counts = np.bincount(labels)  # [n_neg, n_pos]
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,   num_workers=4, pin_memory=pin)

    # 模型与优化器
    model = CNNWithAttention(width_mult=WIDTH_MULT, fc_dim=FC_DIM, attn_dim=ATTN_DIM)
    n_params = count_params(model)
    log(f"CNN+Attn params: {n_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_w = torch.tensor([class_counts[0] / (class_counts[1] + 1e-6)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    early = EarlyStopper(patience=8)
    best_auc = -1.0

    # 训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        metrics = evaluate(model, val_loader, device)
        log(f"Epoch {epoch:02d} | val_acc={metrics['accuracy']:.4f} "
            f"val_f1={metrics['f1']:.4f} val_auc={metrics['auc']:.4f}")
        val_auc = metrics["auc"] if not np.isnan(metrics["auc"]) else -1e-9
        scheduler.step(val_auc)

        # 保存最好
        if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({
                "model": model.state_dict(),
                "width_mult": WIDTH_MULT, "fc_dim": FC_DIM, "attn_dim": ATTN_DIM,
                "n_params": n_params
            }, CKPT_PATH)

        if early.step(val_auc):
            log("Early stopped."); break

    log(f"Best ckpt: {os.path.abspath(CKPT_PATH)} (best val AUC={best_auc:.4f})")

    # 评估（Val/Test）
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    for split, loader in [("Val", val_loader), ("Test", test_loader)]:
        m = evaluate(model, loader, device)
        line = f"[CNN+Attn] {split}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}"
        log(line)
        append_csv(RESULTS_CSV, {
            "model": "cnn_attn",
            "split": split,
            "accuracy": f"{m['accuracy']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "auc": f"{m['auc']:.4f}",
        })

    log(f"Results appended to: {os.path.abspath(RESULTS_CSV)}")
    # 自动打开结果与日志（Windows）
    open_if_windows(RESULTS_CSV)
    open_if_windows(LOG_PATH)

if __name__ == "__main__":
    main()
