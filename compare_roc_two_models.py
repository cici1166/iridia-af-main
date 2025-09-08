# compare_roc_two_models.py
import argparse
import torch
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from common.model_v2 import AFPredictionModelV2
from common.model_cnn import PureCNN

# ===== Dataset类 =====
class ECGDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        self.h5_path = h5_path
        self.samples = pd.read_csv(csv_path)
        self.h5_file = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        row = self.samples.iloc[idx]
        ecg = self.h5_file[row['sample_id']][:]
        if ecg.ndim == 2:
            ecg = ecg[:, 0]  # 取第0导联
        ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # (1, L)
        label = 1 if row['segment_label'] == 'pre_af' else 0
        return ecg, label

# ===== 推理收集概率 =====
@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs_all, labels_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze()
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs_all.extend(prob)
        labels_all.extend(yb.numpy())
    return np.array(probs_all), np.array(labels_all)

# ===== 主函数 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="dataset/processed_data.h5")
    ap.add_argument("--csv", default="splits/test.csv", help="评估用CSV路径")
    ap.add_argument("--bilstm_ckpt", default="best_model_v2.pt")
    ap.add_argument("--cnn_ckpt", default="best_model_cnn.pt")
    ap.add_argument("--out", default="roc_two_models.png")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    ds = ECGDataset(args.h5, args.csv)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    # 加载 CNN-BiLSTM
    model_bilstm = AFPredictionModelV2().to(device)
    ckpt_bi = torch.load(args.bilstm_ckpt, map_location=device,weights_only=False)
    state_bi = ckpt_bi["model_state_dict"] if "model_state_dict" in ckpt_bi else ckpt_bi["model"]
    model_bilstm.load_state_dict(state_bi, strict=True)

    # 加载 Pure CNN
    model_cnn = PureCNN().to(device)
    ckpt_cnn = torch.load(args.cnn_ckpt, map_location=device,weights_only=False)
    state_cnn = ckpt_cnn["model"] if "model" in ckpt_cnn else ckpt_cnn
    model_cnn.load_state_dict(state_cnn, strict=True)

    # 收集概率
    probs_bi, labels_bi = collect_probs(model_bilstm, loader, device)
    probs_cnn, labels_cnn = collect_probs(model_cnn, loader, device)

    # 计算 ROC 曲线
    fpr_bi, tpr_bi, _ = roc_curve(labels_bi, probs_bi)
    auc_bi = auc(fpr_bi, tpr_bi)
    fpr_cnn, tpr_cnn, _ = roc_curve(labels_cnn, probs_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    # 绘制
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_bi, tpr_bi, label=f'CNN-BiLSTM (AUC = {auc_bi:.3f})')
    plt.plot(fpr_cnn, tpr_cnn, label=f'Pure CNN (AUC = {auc_cnn:.3f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"[Saved] ROC 对比图已保存: {args.out}")

if __name__ == "__main__":
    main()
