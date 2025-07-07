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
def save_images(y, x_rec, save_pth):
    max = y.data.max()
    min = y.data.min()
    os.makedirs(save_pth, exist_ok=True)
    imgs_sample = (y.data - min) / (max - min)
    filename = os.path.join(save_pth, "raw.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    imgs_sample = (x_rec.data - min) / (max - min)
    filename = os.path.join(save_pth, "rec.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)


    os.makedirs(os.path.join(save_pth, "all_imgs_raw"), exist_ok=True)
    imgs_sample = (y.data - min) / (max - min)
    for i in range(5):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_raw/{i}.jpg"), nrow=1)

    os.makedirs(os.path.join(save_pth, "all_imgs_rec"), exist_ok=True)
    imgs_sample = (x_rec.data - min) / (max - min)
    for i in range(5):
        img = imgs_sample[i]
        torchvision.utils.save_image(img, os.path.join(save_pth, f"all_imgs_rec/{i}.jpg"), nrow=1)

def Image_evaluate(x,x_):
    def psnr_loss(mse,PIXEL_MAX=1):
        if mse < 1.0e-10:
           return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    ssim_loss = SSIM()
    psnr = psnr_loss(F.mse_loss(x, x_))
    ssim = ssim_loss(x,x_)
    return psnr, torch.mean(ssim)

def compute_rmse(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "张量形状必须相同"
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()


# training based on MTL
def MultiTaskLearning(model, TrainLoader, TestLoader):
    checkpoint_path = os.path.join(cfg.checkpoints_dir)
    os.makedirs(checkpoint_path, exist_ok=True)

    # define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_delay)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.1, verbose=True)
    model, opt, TrainLoader, scheduler = accelerator.prepare(model, opt, TrainLoader,scheduler)


    # define loss function
    Mseloss = nn.MSELoss()
    l1loss = nn.L1Loss()
    # training
    Loss_record = []
    Test_record = []
    for epoch in range(cfg.training_epoch):
        start = time.time()
        model.train()
        epoch_loss = []
        for batch in TrainLoader:
            opt.zero_grad()
            RawImg, signal, sensing_img, distance, angle, rate = batch
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
            print(f"epoch {epoch} | multi-task loss: {multitaskloss.item()} | image reconstruction loss: {loss1.item()} | angle loss: {loss2.item()} | distance loss: {loss3.item()} | rate loss: {loss4.item()}")
            epoch_loss.append(multitaskloss.item())
        # scheduler.step()
        if accelerator.is_main_process:
            save_images(sensing_img, p_sensing_img, os.path.join(cfg.logs_dir, "SensingImages"))
            # save_weights
            torch.save(model.module.state_dict(), os.path.join(checkpoint_path, "SIMAC_AWGN.pth"))
            Loss_record.append(np.mean(epoch_loss))

        if epoch%10==0:
            psnr_res = []
            ssim_res = []
            angle_rmse_res = []
            dis_rmse_res = []
            rate_rmse_res = []

            TestLoader = accelerator.prepare(TestLoader)
            with torch.no_grad():
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


                    dis_rmse = compute_rmse(distance, p_distance)
                    angle_rmse = compute_rmse(angle, p_angle)
                    rate_rmse = compute_rmse(rate, p_rate)

                    psnr_res.append(psnr)
                    ssim_res.append(ssim.item())

                    dis_rmse_res.append(dis_rmse)
                    angle_rmse_res.append(angle_rmse)
                    rate_rmse_res.append(rate_rmse)

            test_res = {
                "PSNR":np.mean(psnr_res),
                "SSIM":np.mean(ssim_res),
                "dis_rmse": np.mean(dis_rmse_res),
                "angle_rmse": np.mean(angle_rmse_res),
                "rate_rmse": np.mean(rate_rmse_res)
            }
            print(f"epoch {epoch} | {test_res}")
            Test_record.append(test_res)

            records = {"train_loss":Loss_record,"Test res":Test_record}
            with open(os.path.join(cfg.logs_dir, "loss.json"), "w",
                      encoding="utf-8")as f:
                f.write(json.dumps(records, ensure_ascii=False, indent=4))

        print("waste time:",time.time()-start)


class Config():
    batch_size = 48
    training_epoch = 50
    SNR_MAX = 26
    SNR_MIN = 0
    Modulations = ["QPSK", "BPSK", "8PSK", "16QAM"]
    lr = 1e-4
    weight_delay = 1e-6
    device = "cuda"
    checkpoints_dir = "checkpoints"
    logs_dir = f"logs"
    dataset_path = r"/Data/ybpeng/Park/result_seg/"

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
    # checkpoint = torch.load("checkpoints/SIMAC_l1.pth", map_location='cpu', weights_only=True)
    # model.load_state_dict(checkpoint,strict=False)
    MultiTaskLearning(model, TrainLoader, TestLoader)
    # cmd run "accelerate launch --config_file accelerate_config.yaml Train.py"





