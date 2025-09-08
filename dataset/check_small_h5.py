import h5py
import os

# 检查小样本文件
script_dir = os.path.dirname(os.path.abspath(__file__))
small_file = os.path.join(script_dir, "ecg_small_sample.h5")

print("=== 检查小样本文件 ===")
print(f"文件路径: {small_file}")
print(f"文件存在: {os.path.exists(small_file)}")

with h5py.File(small_file, "r") as f:
    print(f"\n数据集:")
    for key in f.keys():
        data = f[key]
        print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
    
    print(f"\n属性:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")

print("=== 检查完成 ===")
