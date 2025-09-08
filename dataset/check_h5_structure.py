import h5py
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "processed_data.h5")

print(f"Checking file: {input_file}")
print(f"File exists: {os.path.exists(input_file)}")

def print_h5_structure(path):
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {obj.name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"[GROUP]   {obj.name}")
    
    with h5py.File(path, "r") as f:
        print("=== H5 structure (groups & datasets) ===")
        f.visititems(walk)
        print("=== End of structure ===")

print_h5_structure(input_file) 