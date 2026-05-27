import json
import os
import torch, torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import random
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.datasets as dset

imagenet_mean = np.array([0.4802, 0.4481, 0.3975])
imagenet_std = np.array([0.2770, 0.2691, 0.2821])


import torch

def normalize_complex_tensor(tensor):
    """
    对复数 tensor 进行归一化。

    Args:
        tensor: torch.Tensor，形状为 (6000, 10)，包含复数的 tensor。

    Returns:
        torch.Tensor，归一化后的 tensor。
    """
    # 计算模值
    magnitude = torch.abs(tensor)

    # 避免除以零
    epsilon = 1e-8

    # 按元素归一化（单位化）
    tensor_unit = tensor / (magnitude + epsilon)

    # 按列归一化到 [0, 1]
    col_min = magnitude.min(dim=0, keepdim=True).values
    col_max = magnitude.max(dim=0, keepdim=True).values

    magnitude_normalized = (magnitude - col_min) / (col_max - col_min + epsilon)

    # 结合单位化方向和归一化幅值
    tensor_normalized = tensor_unit * magnitude_normalized

    return tensor_normalized


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.RawImageTr = self.transform1()
        self.SensingImageTr = self.transform2()


    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        img_path = self.data[item]
        SensingImg = Image.open(img_path).convert('RGB')
        SensingImg = self.SensingImageTr(SensingImg)
        # label
        result_path = img_path.replace("result_seg", "result")
        directory, filename = os.path.split(result_path)
        filename, ST_index = os.path.basename(img_path).replace(".jpg", "").rsplit("_", 1)#filename, ST_index = filename.replace(".jpg", "").split("_")
        RawImage_path = os.path.join(directory, filename + ".jpg")
        RawImg = Image.open(RawImage_path).convert('RGB')
        RawImg = self.RawImageTr(RawImg)

        label_path = os.path.join(directory, filename + ".json")
        with open(label_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        box = content[int(ST_index)]
        distance = torch.Tensor([box["distance"]])
        angle = torch.Tensor([box["angle"]])
        rate = torch.Tensor([box["rate"]])
        # cls = torch.Tensor([1 if box["class"]=="car" else 0]).long()
        # cls = cls.squeeze(0)
        SigPath = img_path.replace("result_seg", "result").replace(".jpg", ".npy")
        signal = np.load(SigPath)
        signal = torch.from_numpy(signal).to(torch.complex64)
        # signal = signal.unsqueeze(0)
        # signal = normalize_complex_tensor(signal)

        return RawImg, signal, SensingImg, distance, angle, rate

    def transform1(self):
        compose = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
        return transforms.Compose(compose)

    def transform2(self):
        compose = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        return transforms.Compose(compose)

def get_dataloader(DataPath, BatchSize):
    data = dset.ImageFolder(root=DataPath).imgs
    same_seeds(2024)
    data = [x[0] for i,x in enumerate(data) if i%8==0]
    random.shuffle(data)
    data_size = len(data)
    train_data = data[:int(data_size*0.9)]
    test_data = data[int(data_size*0.9):]

    train_loader = torch.utils.data.DataLoader(CustomDataset(train_data), batch_size=BatchSize,
                                               shuffle=False,pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(CustomDataset(test_data), batch_size=BatchSize,
                                               shuffle=False,pin_memory=True, num_workers=2)


    return train_loader, test_loader

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

if __name__ == '__main__':
    DataPath = "/Data/ybpeng/Park/result_seg"
    BatchSize = 64
    train_loader, test_loader = get_dataloader(DataPath,BatchSize)
    for batch in train_loader:
        RawImg, signal, sensing_img, distance, angle, rate = batch
