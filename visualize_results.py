import os
from PIL import Image, ImageDraw, ImageFont

img_dir = r"E:\Research\SIMAC\logs\SensingImages"

# 选你想对比的epoch
epochs = [0, 10, 20, 30, 40, 49]

raw_path = os.path.join(img_dir, "raw_epoch_49.jpg")

imgs = []
labels = []

raw = Image.open(raw_path).convert("RGB").resize((224, 224))
imgs.append(raw)
labels.append("GT")

for e in epochs:
    rec_path = os.path.join(img_dir, f"rec_epoch_{e}.jpg")
    if os.path.exists(rec_path):
        img = Image.open(rec_path).convert("RGB").resize((224, 224))
        imgs.append(img)
        labels.append(f"Epoch {e}")

w, h = 224, 224
canvas = Image.new("RGB", (w * len(imgs), h + 30), "white")
draw = ImageDraw.Draw(canvas)

for i, (img, label) in enumerate(zip(imgs, labels)):
    canvas.paste(img, (i * w, 30))
    draw.text((i * w + 70, 8), label, fill=(0, 0, 0))

out_path = "GT_vs_Reconstructed.png"
canvas.save(out_path)
print("saved:", out_path)