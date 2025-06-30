import os
import pandas as pd

def get_global_labels_from_csv(records_root, af_threshold=6000):
    """
    为每条记录生成全局标签（0/1）：
    - 如果该记录中存在至少一个 AF 事件，且 af_duration >= af_threshold → 标签为 1；
    - 否则 → 标签为 0。
    
    参数：
        records_root: 所有 record_xxx 文件夹的根目录
        af_threshold: 判定为"有效 AF"的最小持续长度（单位：采样点，默认 6000 = 30秒 @ 200Hz）

    返回：
        DataFrame，包含 record_id 和 label 两列
    """
    record_labels = []

    for record_name in os.listdir(records_root):
        record_path = os.path.join(records_root, record_name)
        
        if os.path.isdir(record_path):
            label_file_csv = os.path.join(record_path, f"{record_name}_ecg_labels.csv")
            
            if os.path.exists(label_file_csv):
                try:
                    # 读取 CSV 标签文件
                    df = pd.read_csv(label_file_csv)

                    # 判断是否存在 AF 持续时间大于等于阈值
                    if 'af_duration' in df.columns:
                        has_af = (df['af_duration'] >= af_threshold).any()
                    else:
                        raise ValueError(f"{label_file_csv} 缺少 'af_duration' 列")

                    label = 1 if has_af else 0
                    record_labels.append({'record_id': record_name, 'label': label})

                except Exception as e:
                    print(f"❌ 处理失败 {label_file_csv}: {e}")

    # 汇总成表格返回
    return pd.DataFrame(record_labels)

# 使用示例
if __name__ == "__main__":
    records_root = "data/records"  # 你的 records 路径
    labels_df = get_global_labels_from_csv(records_root)
    
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath(os.getcwd())
    output_file = os.path.join(current_dir, "global_record_labels.csv")
    
    # 保存文件
    labels_df.to_csv(output_file, index=False)
    print(f"✅ 全部处理完成，标签表已保存为：{output_file}")
    print(f"当前工作目录：{current_dir}")
    print(labels_df)
