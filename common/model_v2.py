import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule(nn.Module):
    """
    CNN模块：用于提取ECG信号的局部特征
    输入：单导联 ECG 信号段（[B, 1, 120000]）
    输出：特征序列（[B, 128, L]）
    """
    def __init__(self):
        super(CNNModule, self).__init__()
        self.cnn = nn.Sequential(
            # 第一层：降采样到1/4
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 第二层：降采样到1/8
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 第三层：降采样到1/16
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 第四层：降采样到1/32
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        # 输入: [B, 1, 120000] → 输出: [B, 128, 3750]
        return self.cnn(x)

class LSTMModule(nn.Module):
    """
    LSTM模块：用于捕捉时序依赖
    输入：CNN提取的特征序列（[B, 128, L]）
    输出：时序特征（[B, 256]）
    """
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, dropout=0.5):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
    def forward(self, x):
        # 输入: [B, 128, L] → [B, L, 128]
        x = x.transpose(1, 2)
        
        # LSTM处理: [B, L, 128] → [B, L, 256]
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出: [B, 256]
        return lstm_out[:, -1, :]

class ClassifierModule(nn.Module):
    """
    分类器模块
    输入：LSTM提取的时序特征（[B, 256]）
    输出：Logit分数（[B, 1]）
    """
    def __init__(self, input_size=256, hidden_size=128, dropout=0.5):
        super(ClassifierModule, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.classifier(x)

class AFPredictionModelV2(nn.Module):
    """
    改进的房颤预测模型（CNN+LSTM混合）
    输入：单导联 ECG 信号段（[B, 1, 120000]）
    输出：房颤预测概率（[B, 1]）
    """
    def __init__(self):
        super(AFPredictionModelV2, self).__init__()
        self.cnn = CNNModule()  # CNN特征提取
        self.lstm = LSTMModule()  # LSTM时序建模
        self.classifier = ClassifierModule()  # 分类器
        
    def forward(self, x):
        # 输入: [B, 1, 120000]
        features = self.cnn(x)  # [B, 128, 3750]
        temporal_features = self.lstm(features)  # [B, 256]
        logits = self.classifier(temporal_features)  # [B, 1]
        return logits

# 用于测试模型是否结构正确
if __name__ == '__main__':
    model = AFPredictionModelV2()
    dummy_input = torch.randn(4, 1, 120000)  # 模拟一个batch
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应输出 (4, 1)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 