import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from datasets.dataset import PreloadedDataset


def MNIST(
        root, 
        split, 
        val_ratio=0.1,
        normalize=True,
        transform=None, 
        dtype='float32',
        device='cpu', 
        download=True
    ):
    # Load data
    assert split in ['train', 'val', 'test']

    train = split in ['train', 'val'] # True for train and val, False for test
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) if normalize else transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    if split == 'train':
        # Build train dataset
        n_train = int(len(dataset) * (1 - val_ratio))
        dataset = torch.utils.data.Subset(dataset, range(0, n_train))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True)
    
    elif split == 'val':
        # Build val dataset
        n_val = len(dataset) * val_ratio
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - n_val, len(dataset)))
        dataset = PreloadedDataset.from_dataset(dataset, transforms.ToTensor(), device, use_tqdm=True)
    
    elif split == 'test':
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True)

    dataset = dataset.to_dtype(dtype).to(device)

    return dataset