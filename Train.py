import torch.optim.lr_scheduler
from model import SIMAC
import random
from torch import nn
import torchvision
import os
from DataUtils import get_dataloader
import json
from accelerate import Accelerator
import math
import time
from ssim import *

# torch.cuda.set_device(0)

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# show image and save
def save_images(y, x_rec, save_pth, epoch):
    max_val = y.data.max()
    min_val = y.data.min()
    os.makedirs(save_pth, exist_ok=True)
    imgs_sample = (y.data - min_val) / (max_val - min_val)
    filename = os.path.join(save_pth, f"raw_epoch_{epoch}.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    imgs_sample = (x_rec.data - min_val) / (max_val - min_val)
    filename = os.path.join(save_pth, f"rec_epoch_{epoch}.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)


    os.makedirs(os.path.join(save_pth, "all_imgs_raw"), exist_ok=True)
    imgs_sample = (y.data - min_val) / (max_val - min_val)
    for i in range(min(5,imgs_sample.shape[0])):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_raw/epoch_{epoch}_{i}.jpg"), nrow=1)

    os.makedirs(os.path.join(save_pth, "all_imgs_rec"), exist_ok=True)
    imgs_sample = (x_rec.data - min_val) / (max_val - min_val)
    for i in range(min(5,imgs_sample.shape[0])):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_rec/epoch_{epoch}_{i}.jpg"), nrow=1)

def Image_evaluate(x,x_):
    def psnr_loss(mse,PIXEL_MAX=1):#图像恢复质量
        if mse < 1.0e-10:
           return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    ssim_loss = SSIM()#图像结构相似度更符合人眼感知
    psnr = psnr_loss(F.mse_loss(x, x_))
    ssim = ssim_loss(x,x_)
    return psnr, torch.mean(ssim)

def compute_rmse(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "张量形状必须相同"
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


# training based on MTL
def MultiTaskLearning(model, TrainLoader, TestLoader,Loss_record, Test_record):
    checkpoint_path = os.path.join(cfg.checkpoints_dir)
    os.makedirs(checkpoint_path, exist_ok=True)

    # define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_delay)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.1, verbose=True)#学习率调度器
    model, opt, TrainLoader, scheduler = accelerator.prepare(model, opt, TrainLoader,scheduler)#自动处理GPU训练


    # define loss function
    Mseloss = nn.MSELoss()
    l1loss = nn.L1Loss()
    # training
    for epoch in range(start_epoch, cfg.training_epoch):
        start = time.time()
        model.train()
        epoch_loss = []
        for batch in TrainLoader:
            opt.zero_grad()
            RawImg, signal, sensing_img, distance, angle, rate = batch#SIMAC输入的东西
            RawImg = RawImg.to(cfg.device)
            signal = signal.to(cfg.device)
            sensing_img = sensing_img.to(cfg.device)
            distance = distance.to(cfg.device)
            angle = angle.to(cfg.device)
            rate = rate.to(cfg.device)
            snr = random.randint(cfg.SNR_MIN,cfg.SNR_MAX)
            if snr < 10:
                modulator = "BPSK"
            elif snr < 18:
                modulator = "QPSK"
            elif snr < 22:
                modulator = "8PSK"
            else:
                modulator = "16QAM"
            # modulator = random.choice(cfg.Modulations)
            ChannelInfo = [f"the SNR is {snr} dB, the signal modulation is {modulator}"]*RawImg.shape[0]
            # no LSE
            # ChannelInfo = [" "]*RawImg.shape[0]


            p_sensing_img, p_angle, p_distance, p_rate = model(signal, RawImg, modulator, snr, ChannelInfo, device=cfg.device)

            # loss 计算
            loss1 = l1loss(p_sensing_img,sensing_img)*100
            loss2 = Mseloss(p_angle,angle)
            loss3 = Mseloss(p_distance,distance)
            loss4 = Mseloss(p_rate,rate)
            multitaskloss = loss1 + loss2 + loss3 + loss4
            multitaskloss.backward()
            opt.step()
            epoch_loss.append(multitaskloss.item())
            if len(epoch_loss) % 10 == 0:
                print(f"epoch {epoch} | multi-task loss: {multitaskloss.item()} | image reconstruction loss: {loss1.item()} | angle loss: {loss2.item()} | distance loss: {loss3.item()} | rate loss: {loss4.item()}")
        # =========================
        # train loss record
        # 记录当前epoch所有batch的平均loss
        # 用于绘制training loss curve
        # =========================   
        Loss_record.append(np.mean(epoch_loss))
        # =========================
        # test every epoch
        # 每个epoch进行一次测试
        # 计算PSNR / SSIM / RMSE
        # =========================
        if (epoch + 1) % 1 == 0:        #epoch % 10 == 0:
            psnr_res = []
            ssim_res = []
            angle_rmse_res = []
            dis_rmse_res = []
            rate_rmse_res = []

            TestLoader = accelerator.prepare(TestLoader)
            with torch.no_grad():
                # switch to evaluation mode
                model.eval()
                for batch in TestLoader:
                    RawImg, signal, sensing_img, distance, angle, rate = batch
                    RawImg = RawImg.to(cfg.device)
                    signal = signal.to(cfg.device)
                    sensing_img = sensing_img.to(cfg.device)
                    distance = distance.to(cfg.device)
                    angle = angle.to(cfg.device)
                    rate = rate.to(cfg.device)
                    snr = random.randint(cfg.SNR_MIN, cfg.SNR_MAX)
                    if snr < 10:
                        modulator = "BPSK"
                    elif snr < 18:
                        modulator = "QPSK"
                    elif snr < 22:
                        modulator = "8PSK"
                    else:
                        modulator = "16QAM"
                    ChannelInfo = [f"The SNR is {snr} dB, the signal modulation is {modulator}"]*RawImg.shape[0]
                    p_sensing_img, p_angle, p_distance, p_rate = model(signal, RawImg, modulator, snr, ChannelInfo, device=cfg.device)
                    # loss 计算
                    max_val = sensing_img.data.max()
                    min_val = sensing_img.data.min()
                    x_norm = (sensing_img.data - min_val) / (max_val - min_val)
                    x_r_norm = (p_sensing_img.data - min_val) / (max_val - min_val)
                    psnr, ssim = Image_evaluate(x_norm,x_r_norm)


                    dis_rmse = compute_rmse(distance.view_as(p_distance), p_distance)
                    angle_rmse = compute_rmse(angle.view_as(p_angle), p_angle)
                    rate_rmse = compute_rmse(rate.view_as(p_rate), p_rate)

                    psnr_res.append(psnr)
                    ssim_res.append(ssim.item())

                    dis_rmse_res.append(dis_rmse)
                    angle_rmse_res.append(angle_rmse)
                    rate_rmse_res.append(rate_rmse)
            # =========================
            # average metrics of current epoch
            # 当前epoch测试指标平均值
            # =========================
            test_res = {
                "PSNR":float(np.mean(psnr_res)),
                "SSIM":float(np.mean(ssim_res)),
                "dis_rmse": float(np.mean(dis_rmse_res)),
                "angle_rmse": float(np.mean(angle_rmse_res)),
                "rate_rmse": float(np.mean(rate_rmse_res))
            }
            print(f"epoch {epoch} | {test_res}")
            save_images(sensing_img, p_sensing_img, os.path.join(cfg.logs_dir, "SensingImages"),epoch)
            # =========================
            # save all test metrics
            # 用于后续绘制PSNR/SSIM/RMSE曲线
            # =========================
            Test_record.append(test_res)
            # =========================
            # save training logs to json
            # 保存loss和测试指标
            # 用于后续曲线可视化
            # =========================
            records = {"train_loss": [float(x) for x in Loss_record],
                       "Test res":[ 
                           {k: float(v) for k, v in res.items()}
                            for res in Test_record
                            ]
                        }
            with open(os.path.join(cfg.logs_dir, "loss.json"), "w",
                      encoding="utf-8")as f:
                f.write(json.dumps(records, ensure_ascii=False, indent=4))
            if accelerator.is_main_process:
            # =========================
            # save checkpoint
            # 保存模型权重和训练状态
            #
            # epoch:
            #     当前训练轮数
            #
            # model_state_dict:
            #     神经网络参数（模型权重）
            #
            # optimizer_state_dict:
            #     优化器状态（学习率、动量等）
            #
            # Loss_record:
            #     所有epoch的训练loss
            #
            # Test_record:
            #     所有epoch的测试指标
            # =========================
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "Loss_record": Loss_record,
                    "Test_record": Test_record,
                    }, os.path.join(checkpoint_path, "latest.pth"))

        print("waste time:",time.time()-start)


class Config():
    batch_size = 1
    training_epoch = 60
    SNR_MAX = 26
    SNR_MIN = 0
    Modulations = ["QPSK", "BPSK", "8PSK", "16QAM"]
    lr = 1e-4
    weight_delay = 1e-6
    device = "cuda"
    checkpoints_dir = "checkpoints"
    logs_dir = f"logs"
    dataset_path = r"E:\Research\SIMAC\real_data\result_seg"

if __name__ == '__main__':
    # hyparametes set
    same_seeds(2048)
    cfg = Config()
    # prepare model
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    cfg.device = accelerator.device
    # prepare data
    TrainLoader, TestLoader = get_dataloader(cfg.dataset_path, cfg.batch_size)

    # train SIMAC
    print("train SIMAC...")
    model = SIMAC().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_delay)
    checkpoint_path = "./checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    resume_path = os.path.join(checkpoint_path, "latest.pth")
    start_epoch = 0

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        Loss_record = checkpoint.get("Loss_record", [])
        Test_record = checkpoint.get("Test_record", [])
        print(f"Resume from epoch {start_epoch}")
    else:
        Loss_record = []
        Test_record = []
    # checkpoint = torch.load("checkpoints/SIMAC_l1.pth", map_location='cpu', weights_only=True)
    # model.load_state_dict(checkpoint,strict=False)
    MultiTaskLearning(model, TrainLoader, TestLoader, Loss_record, Test_record)
    # cmd run "accelerate launch --config_file accelerate_config.yaml Train.py"





