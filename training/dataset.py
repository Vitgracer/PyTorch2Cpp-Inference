import os
import torch
import torchvision
from pathlib import Path
import torchvision.transforms as transforms


def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_path = Path(os.path.dirname(__file__)) / 'data'
    data_path.mkdir(parents=True, exist_ok=True)

    # Load training and validation datasets
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

    return trainloader, valloader