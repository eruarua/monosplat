import json
import numpy as np

pose_path = "/data/baosongze/workspace/3D/3D_debug_data/nersemble_data_expand/train/031_EXP-1-head/pose/cam_param.json"
with open(pose_path, 'r') as f:
    pose_data = json.load(f)

# Check a sample view
view_name = '222200037'
w2c = pose_data['world_2_cam'][view_name]
print(f"Type of w2c: {type(w2c)}")
print(f"Length of w2c: {len(w2c)}")
print(f"First element type: {type(w2c[0])}")
print(f"First element length: {len(w2c[0])}")
print(f"w2c[0][0]: {w2c[0][0]}")

# Convert to numpy
w2c_np = np.array(w2c, dtype=np.float64)
print(f"\nNumPy shape: {w2c_np.shape}")
print(f"NumPy array:\n{w2c_np}")

# Check intrinsics
intr = pose_data['intrinsics']
intr_np = np.array(intr, dtype=np.float64)
print(f"\nIntrinsics shape: {intr_np.shape}")
print(f"Intrinsics:\n{intr_np}")