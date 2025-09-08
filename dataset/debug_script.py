import h5py
import numpy as np
import os
import random

print("=== 调试脚本开始 ===")

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "processed_data.h5")
output_file = os.path.join(script_dir, "ecg_small_sample.h5")

print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")
print(f"输入文件存在: {os.path.exists(input_file)}")

# 检查H5文件结构
with h5py.File(input_file, "r") as f:
    print(f"\n=== H5文件结构 ===")
    print(f"顶层键: {list(f.keys())}")
    
    # 统计pre_af和nsr的数量
    pre_af_count = 0
    nsr_count = 0
    pre_af_keys = []
    nsr_keys = []
    
    for key in f.keys():
        if 'pre_af' in key.lower():
            pre_af_count += 1
            pre_af_keys.append(key)
        elif 'nsr' in key.lower():
            nsr_count += 1
            nsr_keys.append(key)
    
    print(f"pre_af数据集数量: {pre_af_count}")
    print(f"nsr数据集数量: {nsr_count}")
    
    if pre_af_keys:
        print(f"前5个pre_af键: {pre_af_keys[:5]}")
    if nsr_keys:
        print(f"前5个nsr键: {nsr_keys[:5]}")
    
    # 随机选择样本
    n_samples = 3
    if pre_af_keys:
        selected_pre_af = random.sample(pre_af_keys, min(n_samples, len(pre_af_keys)))
        print(f"选择的pre_af: {selected_pre_af}")
        
        # 读取数据
        pre_af_data = []
        for key in selected_pre_af:
            data = f[key][:]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
            pre_af_data.append(data)
        
        # 堆叠数据
        pre_af_stacked = np.stack(pre_af_data)
        print(f"pre_af堆叠后shape: {pre_af_stacked.shape}")
    
    if nsr_keys:
        selected_nsr = random.sample(nsr_keys, min(n_samples, len(nsr_keys)))
        print(f"选择的nsr: {selected_nsr}")
        
        # 读取数据
        nsr_data = []
        for key in selected_nsr:
            data = f[key][:]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
            nsr_data.append(data)
        
        # 堆叠数据
        nsr_stacked = np.stack(nsr_data)
        print(f"nsr堆叠后shape: {nsr_stacked.shape}")

# 保存小样本文件
print(f"\n=== 保存文件 ===")
with h5py.File(output_file, "w") as f_out:
    if 'pre_af_stacked' in locals():
        f_out.create_dataset("pre_af", data=pre_af_stacked)
        print(f"✅ 保存pre_af: shape={pre_af_stacked.shape}")
    
    if 'nsr_stacked' in locals():
        f_out.create_dataset("nsr", data=nsr_stacked)
        print(f"✅ 保存nsr: shape={nsr_stacked.shape}")
    
    f_out.attrs["sample_rate"] = 200

print(f"✅ 小样本文件已创建: {output_file}")
print(f"文件存在: {os.path.exists(output_file)}")
print("=== 调试脚本结束 ===") 