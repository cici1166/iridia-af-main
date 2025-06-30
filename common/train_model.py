import sys
sys.path.append('.')
import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from common.model import AFPredictionModel

# 防止 Windows 下中文输出乱码
sys.stdout.reconfigure(encoding='utf-8')

# 自定义 Dataset
class ECGDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        self.h5_path = h5_path
        self.samples = pd.read_csv(csv_path)
        self.h5_file = None  # lazy open

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        row = self.samples.iloc[idx]
        ecg = self.h5_file[row['sample_id']][:]  # shape: (120000,) 或 (120000, 2)
        if ecg.ndim == 2:
            ecg = ecg[:, 0]  # 只取第0列
        ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # (1, 120000)
        label = 1 if row['segment_label'] == 'pre_af' else 0
        return ecg, label

# 训练过程
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for ecg, label in loader:
        ecg, label = ecg.to(device), label.to(device)
        output = model(ecg).squeeze()
        loss = criterion(output, label.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ecg.size(0)
        all_preds.extend(torch.sigmoid(output).detach().cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader.dataset), evaluate_metrics(all_preds, all_labels)

# 验证过程
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ecg, label in loader:
            ecg, label = ecg.to(device), label.to(device)
            output = model(ecg).squeeze()
            loss = criterion(output, label.float())
            total_loss += loss.item() * ecg.size(0)
            all_preds.extend(torch.sigmoid(output).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader.dataset), evaluate_metrics(all_preds, all_labels)

# 评估指标
def evaluate_metrics(pred_probs, labels):
    preds = [1 if p >= 0.5 else 0 for p in pred_probs]
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, pred_probs)
    return {'acc': acc, 'f1': f1, 'auc': auc}

# 主程序
if __name__ == '__main__':
    data_h5 = 'dataset/processed_data.h5'
    train_csv = 'splits/train.csv'
    val_csv = 'splits/val.csv'
    
    batch_size = 8
    num_epochs = 50  # 增加轮数，因为使用了早停
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 10  # 早停耐心值

    # 加载数据
    train_dataset = ECGDataset(data_h5, train_csv)
    val_dataset = ECGDataset(data_h5, val_csv)
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 计算类别权重
    train_labels = [1 if row['segment_label'] == 'pre_af' else 0 
                   for _, row in train_dataset.samples.iterrows()]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    # 创建带权重的采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # 使用采样器
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )

    # 创建模型和优化器
    model = AFPredictionModel().to(device)
    
    # 使用带权重的损失函数
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5
    )

    # 训练循环
    best_auc = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        print(f"训练批次数: {len(train_loader)}，验证批次数: {len(val_loader)}")
        
        train_loss, train_metrics = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # 更新学习率
        scheduler.step(val_metrics['auc'])

        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"New best model saved at epoch {epoch} (AUC={best_auc:.4f}) -> best_model.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 早停
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

