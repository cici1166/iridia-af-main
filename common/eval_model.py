import sys
sys.path.append('.')
import torch
import pandas as pd
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from common.model_v2 import AFPredictionModelV2

print("=== 模型评估开始 ===")

# 1. 定义和训练时一致的 Dataset 类
class ECGDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        print(f"加载数据集: {csv_path}")
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
        ecg = self.h5_file[row['sample_id']][:]  # (120000,) or (120000, 2)
        if ecg.ndim == 2:
            ecg = ecg[:, 0]
        ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # (1, 120000)
        label = 1 if row['segment_label'] == 'pre_af' else 0
        return ecg, label

# 2. 路径配置
data_h5 = 'dataset/processed_data.h5'
test_csv = 'splits/test.csv'
model_path = 'best_model_v2.pt'
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"数据文件: {data_h5}")
print(f"测试集文件: {test_csv}")
print(f"模型文件: {model_path}")
print(f"设备: {device}")

# 检查文件是否存在
if not os.path.exists(data_h5):
    print(f"❌ 数据文件不存在: {data_h5}")
    exit(1)
if not os.path.exists(test_csv):
    print(f"❌ 测试集文件不存在: {test_csv}")
    exit(1)
if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit(1)

print("✅ 所有文件都存在")

# 3. 加载数据
print("\n=== 加载测试数据 ===")
test_dataset = ECGDataset(data_h5, test_csv)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"测试集批次数量: {len(test_loader)}")

# 4. 加载模型
print("\n=== 加载模型 ===")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = AFPredictionModelV2().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"模型加载完成，最佳AUC: {checkpoint.get('best_auc', 'N/A')}")

# 5. 推理并收集预测
print("\n=== 开始推理 ===")
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for i, (ecg, label) in enumerate(test_loader):
        ecg = ecg.to(device)
        outputs = model(ecg).squeeze()
        probs = torch.sigmoid(outputs).cpu()
        preds = (probs >= 0.5).int()
        y_true.extend(label.numpy())
        y_pred.extend(preds.numpy())
        y_probs.extend(probs.numpy())
        
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(test_loader)} 批次")

print(f"推理完成，共处理 {len(y_true)} 个样本")

# 6. 计算评估指标
print("\n=== 计算评估指标 ===")
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print("混淆矩阵:")
print(cm)
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
print(f"准确率: {acc:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC: {auc:.4f}")

print("\n=== 评估完成 ===")

