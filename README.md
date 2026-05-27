# SIMAC Real-Data Reproduction

A reproduction and engineering adaptation of the SIMAC framework using real CARLA-Sionna multimodal wireless data.

---

# Project Highlights

- Reproduced the SIMAC semantic communication framework
- Integrated real CARLA camera data and Sionna CIR channel data
- Built multimodal dataloader and training pipeline
- Implemented checkpoint saving and resume training
- Added visualization and metric evaluation tools

---

# Reconstruction Results

## Ground Truth vs Reconstruction

![GT_vs_Rec](assets/reconstructions/GT_vs_Reconstructed.png)

---

# Training Curves

## Training Loss

![loss](assets/curves/train_loss_curve.png)

## PSNR

![psnr](assets/curves/PSNR_curve.png)

## SSIM

![ssim](assets/curves/SSIM_curve.png)

---

# Sensing Metrics

## Angle RMSE

![angle](assets/curves/angle_rmse_curve.png)

## Distance RMSE

![distance](assets/curves/dis_rmse_curve.png)

## Rate RMSE

![rate](assets/curves/rate_rmse_curve.png)