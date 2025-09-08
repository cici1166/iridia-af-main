import sys
sys.path.append('.')
try:
	# 防止 Windows 下中文输出乱码
	sys.stdout.reconfigure(encoding='utf-8')
except Exception:
	pass
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from common.model_v2 import AFPredictionModelV2
from tqdm import tqdm

# ===== 数据集类（与训练代码一致） =====
class ECGDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        print(f"正在加载数据集: {csv_path}")
        self.h5_path = h5_path
        self.samples = pd.read_csv(csv_path)
        self.h5_file = None
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        row = self.samples.iloc[idx]
        ecg = self.h5_file[row['sample_id']][:]
        if ecg.ndim == 2:
            ecg = ecg[:, 0]
        ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)
        label = 1 if row['segment_label'] == 'pre_af' else 0
        return ecg, label

# ===== 评估函数 =====
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ecg, label in tqdm(loader, desc="Evaluating"):
            ecg, label = ecg.to(device), label.to(device)
            output = model(ecg).squeeze()
            all_preds.extend(torch.sigmoid(output).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds)
    return acc, f1, auc

if __name__ == "__main__":
	try:
		# ==== 配置 ====
		data_h5 = 'dataset/processed_data.h5'
		val_csv = 'splits/val.csv'
		test_csv = 'splits/test.csv'
		batch_size = 16
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print("=== 评估脚本启动 ===")
		print(f"设备: {device}")
		print(f"数据文件: {data_h5}")
		print(f"验证集: {val_csv}")
		print(f"测试集: {test_csv}")

		# ==== 加载模型 ====
		model = AFPredictionModelV2().to(device)
		checkpoint = torch.load('best_model_v2.pt', map_location=device, weights_only=False)
		model.load_state_dict(checkpoint['model_state_dict'])
		print(f"加载最佳模型（AUC={checkpoint.get('best_auc', float('nan')):.4f}）")

		# ==== 加载数据 ====
		val_dataset = ECGDataset(data_h5, val_csv)
		test_dataset = ECGDataset(data_h5, test_csv)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		# ==== 评估 ====
		print("\n=== 验证集结果 ===")
		acc, f1, auc = evaluate(model, val_loader, device)
		print(f"Val - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

		print("\n=== 测试集结果 ===")
		acc, f1, auc = evaluate(model, test_loader, device)
		print(f"Test - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

		print("\n=== 评估完成 ===")
	except Exception as e:
		print("\n评估过程中出错:", str(e))
		import traceback
		traceback.print_exc()
