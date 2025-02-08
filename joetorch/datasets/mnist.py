import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from joetorch.datasets.dataset import PreloadedDataset

def FashionMNIST(root, split, val_ratio=0.1, normalize=False, transform=None, dtype='float32', device='cpu', download=True):
    assert normalize is False, "Normalization is not supported for FashionMNIST"
    return MNIST(root, split, val_ratio, normalize, transform, dtype, device, download, fashion=True)

def MNIST(
        root, 
        split, 
        val_ratio=0.0,
        normalize=True,
        augment=False,
        transform=None, 
        dtype=torch.float32,
        device=torch.device('cpu'),
        download=True,
        fashion=False,
    ):
    # Load data
    assert split in ['train', 'val', 'test']

    train = split in ['train', 'val'] # True for train and val, False for test
    loading_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) if normalize else transforms.ToTensor()
    if fashion:
        dataset = datasets.FashionMNIST(root=root, train=train, transform=loading_transform, download=download)
    else:
        dataset = datasets.MNIST(root=root, train=train, transform=loading_transform, download=download)

    fill_value = dataset[0][0].min().item()

    if transform is None:
        transform = []
    if type(transform) is not list:
        transform = [transform]
    if augment:
        transform = [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, fill=fill_value)] + transform
    if len(transform) > 0:
        transform = transforms.Compose(transform)
    else:
        transform = None

    if split == 'train':
        # Build train dataset
        n_train = int(len(dataset) * (1 - val_ratio))
        dataset = torch.utils.data.Subset(dataset, range(0, n_train))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True)
    
    elif split == 'val':
        # Build val dataset
        n_val = int(len(dataset) * val_ratio)
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - n_val, len(dataset)))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True)
    
    elif split == 'test':
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm=True)

    dataset = dataset.to(dtype).to(device)

    return dataset