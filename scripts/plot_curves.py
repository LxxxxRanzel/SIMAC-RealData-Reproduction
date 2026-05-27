import json
import matplotlib.pyplot as plt

log_path = "logs/loss.json"

with open(log_path, "r", encoding="utf-8") as f:
    records = json.load(f)

train_loss = records["train_loss"]
test_res = records["Test res"]

plt.figure()
plt.plot(train_loss)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss")
plt.savefig("train_loss_curve.png", dpi=300)

metrics = ["PSNR", "SSIM", "dis_rmse", "angle_rmse", "rate_rmse"]

for metric in metrics:
    values = [x[metric] for x in test_res]

    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.savefig(f"{metric}_curve.png", dpi=300)
    plt.close()

print("curves saved.")