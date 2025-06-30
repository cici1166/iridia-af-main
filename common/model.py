import torch
import torch.nn as nn
import torch.nn.functional as F

class DLAModule(nn.Module):
    """
    DL-A: 局部特征提取模块（CNN）
    输入：单导联 ECG 信号段（[B, 1, 120000]）
    输出：固定维度特征向量（[B, 128]）
    """
    def __init__(self):
        super(DLAModule, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        # 输入: [B, 1, 120000] → 输出: [B, 128, 1] → squeeze → [B, 128]
        x = self.cnn(x)
        x = x.squeeze(-1)
        return x

class DLCModule(nn.Module):
    """
    DL-C: 分类器模块
    输入：DL-A 提取的特征向量（[B, 128]）
    输出：Logit 分数（[B, 1]）
    """
    def __init__(self):
        super(DLCModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)  # 输出一个 logit 值
        )

    def forward(self, x):
        return self.fc(x)

class AFPredictionModel(nn.Module):
    """
    房颤预测模型
    输入：单导联 ECG 信号段（[B, 1, 120000]）
    输出：房颤预测概率（[B, 1]）
    """
    def __init__(self):
        super(AFPredictionModel, self).__init__()
        self.dla = DLAModule()  # 特征提取模块
        self.dlc = DLCModule()  # 分类器模块

    def forward(self, x):
        # 输入: [B, 1, 120000] → 输出: [B, 1]
        features = self.dla(x)  # [B, 128]
        logits = self.dlc(features)  # [B, 1]
        return logits

# 用于测试模型是否结构正确
if __name__ == '__main__':
    model = AFPredictionModel()
    dummy_input = torch.randn(4, 1, 120000)  # 模拟一个batch
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (4, 1)

