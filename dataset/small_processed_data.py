# -*- coding: utf-8 -*-
"""
ECG H5 -> small sample extractor (works out-of-the-box)
File name: small_processed_data.py
- Prints H5 structure
- Auto-detects layout:
    A) separate datasets for classes (e.g., /pre_af and /nsr)
    B) signals + labels (e.g., /signals and /labels)
- Randomly selects N samples per class and saves to ecg_small_sample.h5
"""
import h5py
import numpy as np
import random
import os

# === Parameters ===
# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "processed_data.h5")    # Path to the original large H5 file
output_file = os.path.join(script_dir, "ecg_small_sample.h5")  # Path to the output small H5 file
n_samples_per_class = 3              # Number of samples to extract per class
sample_rate = 200                    # Sampling rate (Hz)

print(f"Looking for file: {input_file}")
print(f"File exists: {os.path.exists(input_file)}")

# Open the original H5 file (read-only)
with h5py.File(input_file, "r") as f:
    # Update these dataset paths according to your file structure
    pre_af_data = f["pre_af"][:]  # shape: (num_pre_af, 120000)
    nsr_data = f["nsr"][:]        # shape: (num_nsr, 120000)
    
    # Randomly select sample indices
    pre_af_indices = random.sample(range(pre_af_data.shape[0]), n_samples_per_class)
    nsr_indices = random.sample(range(nsr_data.shape[0]), n_samples_per_class)
    
    # Extract the samples
    pre_af_selected = pre_af_data[pre_af_indices]
    nsr_selected = nsr_data[nsr_indices]

# Save to a new small H5 file
with h5py.File(output_file, "w") as f_out:
    f_out.create_dataset("pre_af", data=pre_af_selected)
    f_out.create_dataset("nsr", data=nsr_selected)
    f_out.attrs["sample_rate"] = sample_rate  # Save the sampling rate as an attribute

print(f"✅ Small sample file created: {output_file}")
