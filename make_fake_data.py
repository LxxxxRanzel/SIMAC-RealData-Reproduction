import os
import json
import numpy as np
from PIL import Image

base = r"E:\Research\SIMAC\fake_data"
seg_dir = os.path.join(base, "result_seg", "class0")
res_dir = os.path.join(base, "result", "class0")

os.makedirs(seg_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

for i in range(10):
    name = f"sample{i}"

    # 感知图像：result_seg/class0/sample0_0.jpg
    seg_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    seg_img.save(os.path.join(seg_dir, f"{name}_0.jpg"))

    # 原始图像：result/class0/sample0.jpg
    raw_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    raw_img.save(os.path.join(res_dir, f"{name}.jpg"))

    # 信号：result/class0/sample0_0.npy
    signal = np.random.randn(10, 6000) + 1j * np.random.randn(10, 6000)
    np.save(os.path.join(res_dir, f"{name}_0.npy"), signal.astype(np.complex64))

    # 标签：result/class0/sample0.json
    label = [{
        "distance": [0.5],
        "angle": [0.3],
        "rate": [0.8]
    }]
    with open(os.path.join(res_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(label, f)

print("fake data created:", base)