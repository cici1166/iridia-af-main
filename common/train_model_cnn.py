import sys
sys.path.append('.')
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.model_cnn import PureCNN, count_params
from common.dataset_h5 import ECGH5Dataset

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import random

from tqdm.auto import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, scan_threshold=False):
    """
    返回 {'accuracy', 'f1', 'auc'}，并在日志里打印预测为正比例 pos_rate。
    当 scan_threshold=True 时，额外扫描阈值（0.05~0.95）报告最大F1与对应阈值。
    """
    model.eval()
    ys, ps = [], []
    pbar = tqdm(loader, desc="[Val]", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.int64)

    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred, zero_division=0)  # 防止除零，极端情况下返回0
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")

    pos_rate = float(pred.mean())
    print(f"[Eval] pos_rate={pos_rate:.3f}")

    if scan_threshold:
        thr_list = np.linspace(0.05, 0.95, 91)
        best_t, best_f1 = 0.5, 0.0
        for t in thr_list:
            f1_t = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        print(f"[Eval] Threshold scan: best_F1={best_f1:.4f} at thr={best_t:.2f}")

    return {"accuracy": acc, "f1": f1, "auc": auc}


class EarlyStopper:
    def __init__(self, patience=8):
        self.patience = patience
        self.best = None
        self.bad = 0
    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad > self.patience


def main():
    ap = argparse.ArgumentParser()
    # 与主训练保持一致的默认路径/超参
    ap.add_argument("--h5", default="dataset/processed_data.h5")
    ap.add_argument("--train_csv", default="splits/train.csv")
    ap.add_argument("--val_csv", default="splits/val.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # 模型可调，用于对齐参数量（±10%）
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--fc_dim", type=int, default=128)

    # 采样策略（如主训练没用加权采样，就不要传 --use_weights）
    ap.add_argument("--use_weights", action="store_true")

    # 可选：BCEWithLogitsLoss 中使用 pos_weight（类不平衡更稳）
    ap.add_argument("--use_pos_weight", action="store_true")

    # 可选：验证阶段扫描阈值，报告最大F1与阈值（不影响返回指标）
    ap.add_argument("--scan_threshold", action="store_true")

    # 输出 checkpoint
    ap.add_argument("--ckpt", default="best_model_cnn.pt")

    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    # 设备与 pin_memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()
    print(f"[Device] Using {device}, pin_memory={pin_mem}")

    # 数据
    train_ds = ECGH5Dataset(args.h5, args.train_csv, normalize=False)
    val_ds   = ECGH5Dataset(args.h5, args.val_csv,   normalize=False)

    # 打印验证集正负比例
    val_labels = [int(y.item()) for _, y in val_ds]
    val_pos = sum(val_labels); val_neg = len(val_labels) - val_pos
    print(f"[Val split] pos={val_pos}, neg={val_neg}, pos_ratio={val_pos/len(val_labels):.3f}")

    # 采样器或普通 DataLoader
    if args.use_weights:
        tr_labels = [int(y.item()) for _, y in train_ds]
        pos = sum(tr_labels); neg = len(tr_labels) - pos
        # 频次反比权重
        w_pos = neg / (pos + 1e-6); w_neg = pos / (neg + 1e-6)
        weights = [w_pos if l == 1 else w_neg for l in tr_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=0, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=0, pin_memory=pin_mem)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=pin_mem)

    # 模型
    model = PureCNN(width_mult=args.width_mult, fc_dim=args.fc_dim)
    n_params = count_params(model)
    print(f"[PureCNN] Trainable params: {n_params:,}")
    if args.dry_run:
        return

    model.to(device)

    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 损失函数：可选 pos_weight（类别不平衡更鲁棒）
    if args.use_pos_weight:
        tr_labels_pw = [int(y.item()) for _, y in train_ds]
        pos = sum(tr_labels_pw); neg = len(tr_labels_pw) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
        print(f"[Loss] Using pos_weight={pos_weight.item():.4f}  (neg={neg}, pos={pos})")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 学习率调度：ReduceLROnPlateau（factor=0.5, patience=4, min_lr=1e-6），监控 val_auc
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    # 早停：监控 val_auc，patience=8
    early = EarlyStopper(patience=8)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)

            # 目标显式 float，必要时对齐形状避免广播问题
            if logits.shape != yb.shape:
                yb = yb.view_as(logits)
            loss = criterion(logits, yb.float())

            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证（带进度条 + pos_rate 打印 + 可选阈值扫描）
        metrics = evaluate(model, val_loader, device, scan_threshold=args.scan_threshold)
        print(f"Epoch {epoch:02d} | val_acc={metrics['accuracy']:.4f} "
              f"val_f1={metrics['f1']:.4f} val_auc={metrics['auc']:.4f}")

        # 调度器步进（监控 val_auc）
        val_auc = metrics["auc"] if not np.isnan(metrics["auc"]) else -1e9
        scheduler.step(val_auc)

        # 保存最好
        if not np.isnan(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {"model": model.state_dict(),
                 "n_params": n_params,
                 "width_mult": args.width_mult,
                 "fc_dim": args.fc_dim},
                args.ckpt
            )

        # 早停
        if early.step(val_auc):
            print("Early stopped.")
            break

    print(f"Best checkpoint saved: {args.ckpt}  (best val AUC={best_auc:.4f})")


if __name__ == "__main__":
    main()
