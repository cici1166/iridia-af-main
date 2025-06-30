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
from common.model_v2 import AFPredictionModelV2
from tqdm import tqdm

# 防止 Windows 下中文输出乱码
sys.stdout.reconfigure(encoding='utf-8')

# 自定义 Dataset
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
        try:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')

            row = self.samples.iloc[idx]
            ecg = self.h5_file[row['sample_id']][:]  # shape: (120000,) 或 (120000, 2)
            if ecg.ndim == 2:
                ecg = ecg[:, 0]  # 只取第0列
            ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)  # (1, 120000)
            label = 1 if row['segment_label'] == 'pre_af' else 0
            return ecg, label
        except Exception as e:
            print(f"加载样本 {idx} 时出错: {str(e)}")
            raise e

# 训练过程
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (ecg, label) in enumerate(pbar):
        try:
            ecg, label = ecg.to(device), label.to(device)
            
            # 前向传播
            output = model(ecg).squeeze()
            loss = criterion(output, label.float())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 记录损失和预测
            total_loss += loss.item() * ecg.size(0)
            all_preds.extend(torch.sigmoid(output).detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"训练批次 {batch_idx} 时出错: {str(e)}")
            raise e

    return total_loss / len(loader.dataset), evaluate_metrics(all_preds, all_labels)

# 验证过程
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for batch_idx, (ecg, label) in enumerate(pbar):
            try:
                ecg, label = ecg.to(device), label.to(device)
                output = model(ecg).squeeze()
                loss = criterion(output, label.float())
                
                total_loss += loss.item() * ecg.size(0)
                all_preds.extend(torch.sigmoid(output).cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"验证批次 {batch_idx} 时出错: {str(e)}")
                raise e

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
    try:
        # 配置参数
        data_h5 = 'dataset/processed_data.h5'
        train_csv = 'splits/train.csv'
        val_csv = 'splits/val.csv'
        
        batch_size = 16
        num_epochs = 50
        lr = 1e-4
        weight_decay = 1e-5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patience = 10

        print("\n=== 配置信息 ===")
        print(f"数据文件: {data_h5}")
        print(f"训练集文件: {train_csv}")
        print(f"验证集文件: {val_csv}")
        print(f"设备: {device}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {lr}")
        print(f"权重衰减: {weight_decay}")
        print(f"早停耐心值: {patience}")

        # 加载数据
        print("\n=== 加载数据 ===")
        train_dataset = ECGDataset(data_h5, train_csv)
        val_dataset = ECGDataset(data_h5, val_csv)

        # 计算类别权重
        print("\n=== 计算类别权重 ===")
        train_labels = [1 if row['segment_label'] == 'pre_af' else 0 
                       for _, row in train_dataset.samples.iterrows()]
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        print(f"类别分布: {dict(zip(range(len(class_counts)), class_counts))}")
        print(f"类别权重: {dict(zip(range(len(class_weights)), class_weights))}")
        
        # 创建带权重的采样器
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )

        # 创建数据加载器
        print("\n=== 创建数据加载器 ===")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # 创建模型
        print("\n=== 创建模型 ===")
        model = AFPredictionModelV2().to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
        
        # 使用带权重的损失函数
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 使用AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
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
        
        print("\n=== 开始训练 ===")
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # 训练和验证
            train_loss, train_metrics = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

            # 打印指标
            print(f"\nEpoch {epoch:02d} 结果:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train - Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

            # 更新学习率
            scheduler.step(val_metrics['auc'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")

            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auc': best_auc,
                }, 'best_model_v2.pt')
                print(f"New best model saved at epoch {epoch} (AUC={best_auc:.4f}) -> best_model_v2.pt")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")
                break

        print("\n训练完成！")
        print(f"最佳模型在 epoch {best_epoch}，AUC: {best_auc:.4f}")

    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc() 