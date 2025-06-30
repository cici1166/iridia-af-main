import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 防止 Windows 下中文输出乱码
sys.stdout.reconfigure(encoding='utf-8')

# 配置比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def main():
    metadata = pd.read_csv('dataset/metadata.csv')
    
    # 计算每个样本的权重（用于分层采样）
    # 对于pre_af样本，权重为1，对于nsr样本，权重为pre_af样本数量的倒数
    pre_af_count = (metadata['segment_label'] == 'pre_af').sum()
    nsr_count = (metadata['segment_label'] == 'nsr').sum()
    sample_weights = np.where(metadata['segment_label'] == 'pre_af', 
                            1.0, 
                            pre_af_count / nsr_count)
    
    # 确定每个 patient 是否含有 pre_af（用于患者级分层）
    label_map = metadata.groupby('patient_id')['segment_label'].apply(
        lambda x: 'pre_af' if 'pre_af' in x.values else 'nsr'
    )
    patient_df = label_map.reset_index().rename(columns={'segment_label': 'patient_label'})
    
    # 计算每个患者的权重
    patient_weights = patient_df['patient_label'].map(
        lambda x: 1.0 if x == 'pre_af' else pre_af_count / nsr_count
    )
    
    # 第一步：按 patient_id 分层划分训练集
    train_ids, temp_ids = train_test_split(
        patient_df,
        stratify=patient_df['patient_label'],
        train_size=TRAIN_RATIO,
        random_state=42
    )
    
    # 第二步：从 temp 中划分验证集和测试集
    # 确保每个集合都包含两种类别的患者，并且比例接近
    pos_temp = temp_ids[temp_ids['patient_label'] == 'pre_af']
    neg_temp = temp_ids[temp_ids['patient_label'] == 'nsr']
    
    # 计算每个集合应该包含的阳性患者数量
    pos_val_size = max(1, int(len(pos_temp) * VAL_RATIO / (VAL_RATIO + TEST_RATIO)))
    pos_test_size = len(pos_temp) - pos_val_size
    
    # 计算每个集合应该包含的阴性患者数量，保持与阳性患者相似的比例
    neg_val_size = max(1, int(pos_val_size * (nsr_count / pre_af_count)))
    neg_test_size = max(1, int(pos_test_size * (nsr_count / pre_af_count)))
    
    # 如果阴性患者数量不足，则按比例减少
    if neg_val_size + neg_test_size > len(neg_temp):
        scale = len(neg_temp) / (neg_val_size + neg_test_size)
        neg_val_size = max(1, int(neg_val_size * scale))
        neg_test_size = len(neg_temp) - neg_val_size
    
    # 划分阳性患者
    if len(pos_temp) > 0:
        val_pos = pos_temp.sample(n=pos_val_size, random_state=42)
        test_pos = pos_temp.drop(val_pos.index)
    else:
        val_pos = pd.DataFrame(columns=pos_temp.columns)
        test_pos = pd.DataFrame(columns=pos_temp.columns)
    
    # 划分阴性患者
    if len(neg_temp) > 0:
        val_neg = neg_temp.sample(n=neg_val_size, random_state=42)
        test_neg = neg_temp.drop(val_neg.index)
    else:
        val_neg = pd.DataFrame(columns=neg_temp.columns)
        test_neg = pd.DataFrame(columns=neg_temp.columns)
    
    # 合并验证集和测试集的患者
    val_ids = pd.concat([val_pos, val_neg], ignore_index=True)
    test_ids = pd.concat([test_pos, test_neg], ignore_index=True)
    
    print("患者划分结果：")
    print(f"训练集患者数：{len(train_ids)}")
    print(f"验证集患者数：{len(val_ids)}")
    print(f"测试集患者数：{len(test_ids)}")
    
    # 映射样本数据
    train_data = metadata[metadata['patient_id'].isin(train_ids['patient_id'])]
    val_data = metadata[metadata['patient_id'].isin(val_ids['patient_id'])]
    test_data = metadata[metadata['patient_id'].isin(test_ids['patient_id'])]
    
    # 保存为 CSV
    os.makedirs('splits', exist_ok=True)
    train_data.to_csv('splits/train.csv', index=False)
    val_data.to_csv('splits/val.csv', index=False)
    test_data.to_csv('splits/test.csv', index=False)
    
    # 打印各集分布
    def print_dist(name, df):
        counts = df['segment_label'].value_counts().to_dict()
        total = len(df)
        pre_af = counts.get('pre_af', 0)
        nsr = counts.get('nsr', 0)
        print(f"\n{name}集样本数：{total}")
        print(f"  pre_af: {pre_af} ({pre_af/total*100:.1f}%)")
        print(f"  nsr   : {nsr} ({nsr/total*100:.1f}%)")
    
    print_dist("训练", train_data)
    print_dist("验证", val_data)
    print_dist("测试", test_data)
    
    print("\n数据划分完成，文件已保存至 splits/ 目录。")

if __name__ == '__main__':
    main()
