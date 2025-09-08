import sys
sys.path.append('.')

import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from common.model_cnn import PureCNN
from common.dataset_h5 import ECGH5Dataset

def evaluate_model(model, loader, device):
    """评估模型性能"""
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算指标
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'probs': all_probs,
        'labels': all_labels
    }

def main():
    parser = argparse.ArgumentParser(description='评估CNN模型')
    parser.add_argument('--ckpt', default='best_model_cnn.pt', help='模型检查点路径')
    parser.add_argument('--h5', default='dataset/processed_data.h5', help='H5数据文件路径')
    parser.add_argument('--val_csv', default='splits/val.csv', help='验证集CSV路径')
    parser.add_argument('--test_csv', default='splits/test.csv', help='测试集CSV路径')
    parser.add_argument('--results_csv', default='results_val_test.csv', help='结果CSV输出路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    
    args = parser.parse_args()
    
    print("=== CNN模型评估 ===")
    print(f"模型文件: {args.ckpt}")
    print(f"数据文件: {args.h5}")
    print(f"验证集: {args.val_csv}")
    print(f"测试集: {args.test_csv}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    try:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = PureCNN(
            width_mult=checkpoint.get('width_mult', 1.0),
            fc_dim=checkpoint.get('fc_dim', 128)
        )
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        print(f"✅ 模型加载成功，参数数量: {checkpoint.get('n_params', 'N/A')}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载数据
    try:
        val_dataset = ECGH5Dataset(args.h5, args.val_csv, normalize=True)
        test_dataset = ECGH5Dataset(args.h5, args.test_csv, normalize=True)
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"✅ 数据加载成功")
        print(f"   验证集: {len(val_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 评估
    print("\n=== 评估结果 ===")
    
    # 验证集评估
    print("验证集评估...")
    val_results = evaluate_model(model, val_loader, device)
    print(f"Val - Accuracy: {val_results['accuracy']:.4f}")
    print(f"Val - F1: {val_results['f1']:.4f}")
    print(f"Val - AUC: {val_results['auc']:.4f}")
    print(f"Val - 混淆矩阵:\n{val_results['confusion_matrix']}")
    
    # 测试集评估
    print("\n测试集评估...")
    test_results = evaluate_model(model, test_loader, device)
    print(f"Test - Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test - F1: {test_results['f1']:.4f}")
    print(f"Test - AUC: {test_results['auc']:.4f}")
    print(f"Test - 混淆矩阵:\n{test_results['confusion_matrix']}")
    
    # 保存结果
    results_df = pd.DataFrame([
        {
            'split': 'Val',
            'accuracy': val_results['accuracy'],
            'f1': val_results['f1'],
            'auc': val_results['auc']
        },
        {
            'split': 'Test',
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'auc': test_results['auc']
        }
    ])
    
    results_df.to_csv(args.results_csv, index=False)
    print(f"\n✅ 结果已保存到: {args.results_csv}")
    print("=== 评估完成 ===")

if __name__ == '__main__':
    main()
