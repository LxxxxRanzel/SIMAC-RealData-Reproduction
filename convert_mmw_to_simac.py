import os
import json
import shutil
import numpy as np
import yaml

sensor_dir = r"E:\Research\SIMAC\data\Town03_Tjunction_wiz_slope_seed42\cav_1"
channel_dir = r"E:\Research\SIMAC\data\Nt_1_16_Nr_1_16_fc_28GHz\Town03\Town03_Tjunction\cav_1"
out_base = r"E:\Research\SIMAC\real_data"

result_dir = os.path.join(out_base, "result", "class0")
seg_dir = os.path.join(out_base, "result_seg", "class0")
os.makedirs(result_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

frames = []
for f in os.listdir(sensor_dir):
    if f.endswith("_camera0.png"):
        frame_id = f.split("_camera0.png")[0]
        if os.path.exists(os.path.join(channel_dir, f"{frame_id}_paths.npz")):
            frames.append(frame_id)

frames = sorted(frames)[:100]

for frame_id in frames:
    # 1. 原图 RawImg
    raw_img = os.path.join(sensor_dir, f"{frame_id}_camera0.png")
    shutil.copy(raw_img, os.path.join(result_dir, f"{frame_id}.jpg"))

    # 2. 感知图 SensingImg，先用同一张图占位
    shutil.copy(raw_img, os.path.join(seg_dir, f"{frame_id}_0.jpg"))

    # 3. 信道 signal
    npz_path = os.path.join(channel_dir, f"{frame_id}_paths.npz")
    data = np.load(npz_path)
    a = data["a"].reshape(-1).astype(np.complex64)

    signal_1d = np.zeros(6000, dtype=np.complex64)
    signal_1d[:min(len(a), 6000)] = a[:6000]
    signal = np.tile(signal_1d[None, :], (10, 1))

    np.save(os.path.join(result_dir, f"{frame_id}_0.npy"), signal)

    # 4. 标签 distance / angle / rate
    yaml_path = os.path.join(sensor_dir, f"{frame_id}.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    # 第一版先给默认值，后面我们再从yaml里精确提取
    label = [
        {
            "distance": [0.5],
            "angle": [0.3],
            "rate": [0.8],
        }
    ]

    with open(os.path.join(result_dir, f"{frame_id}.json"), "w", encoding="utf-8") as f:
        json.dump(label, f)

print("Converted frames:", len(frames))
print("Output:", out_base)