import os
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

dirname = os.path.dirname(os.path.realpath(__file__))


def get_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    traindata = datasets.CIFAR10(root='./CIFAR', train=True, transform=transform, download=True)

    print('Dataset created')

    dataloader = data.DataLoader(
        traindata,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=128)
    print('Dataloader created')

    return dataloader