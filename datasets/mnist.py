import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from datasets.dataset import PreloadedDataset

def FashionMNIST(root, split, val_ratio=0.1, normalize=False, transform=None, dtype='float32', device='cpu', download=True):
    assert normalize is False, "Normalization is not supported for FashionMNIST"
    return MNIST(root, split, val_ratio, normalize, transform, dtype, device, download, fashion=True)

def MNIST(
        root, 
        split, 
        val_ratio=0.1,
        normalize=True,
        augment=False,
        transforms=None, 
        dtype='float32',
        device='cpu', 
        download=True,
        fashion=False,
    ):
    # Load data
    assert split in ['train', 'val', 'test']

    train = split in ['train', 'val'] # True for train and val, False for test
    normalize_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) if normalize else transforms.ToTensor()
    if fashion:
        dataset = datasets.FashionMNIST(root=root, train=train, transform=normalize_transform, download=download)
    else:
        dataset = datasets.MNIST(root=root, train=train, transform=normalize_transform, download=download)

    if transforms is None:
        transforms = []
    if type(transforms) is not list:
        transforms = [transforms]
    if augment:
        transforms = [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)] + transforms
    transform = transforms.Compose(transforms)

    if split == 'train':
        # Build train dataset
        n_train = int(len(dataset) * (1 - val_ratio))
        dataset = torch.utils.data.Subset(dataset, range(0, n_train))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True, augment=augment)
    
    elif split == 'val':
        # Build val dataset
        n_val = len(dataset) * val_ratio
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - n_val, len(dataset)))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True, augment=augment)
    
    elif split == 'test':
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True, augment=augment)

    dataset = dataset.to_dtype(dtype).to(device)

    return dataset