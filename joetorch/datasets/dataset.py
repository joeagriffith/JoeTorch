import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from PIL import Image

def remove_to_tensor(transform):
    if type(transform) == transforms.ToTensor:
        transform = None

    if type(transform) == transforms.Compose:
        new_transforms = []
        for t in transform.transforms:
            if type(t) != transforms.ToTensor:
                new_transforms.append(t)
        transform = transforms.Compose(new_transforms)
    return transform


class PreloadedDataset(Dataset):
    def __init__(self, images=None, transformed_images=None, targets=None, transform=None, classes=None, device=None):
        transform = remove_to_tensor(transform)

        self.images = images
        self.transformed_images = transformed_images
        self.targets = targets
        self.transform = transform
        self.classes = classes
        self.device = device
        if self.classes is None:
            self.classes = torch.unique(self.targets)
        if self.images is not None:
            self.update_transformed_images()

    def from_dataset(dataset, transform, device="cpu", use_tqdm=True):

        data = []
        targets = []
        loop = tqdm(range(len(dataset)), leave=False) if use_tqdm else range(len(dataset))
        for i in loop:
            d, t = dataset.__getitem__(i)
            if type(d) is not torch.Tensor:
                d = torch.tensor(d)
            if type(t) is not torch.Tensor:
                t = torch.tensor(t)
            data.append(d)
            targets.append(t)
        
        images = torch.stack(data).to(device)
        targets = torch.stack(targets).to(device)

        return PreloadedDataset(images=images, targets=targets, transform=transform, device=device)

    def from_tensors(data, targets, transform, device="cpu"):
        assert type(data) == torch.Tensor, "Data must be a torch.Tensor"
        assert type(targets) == torch.Tensor, "Targets must be a torch.Tensor"

        if data.device != device:
            data = data.to(device)
        if targets.device != device:
            targets = targets.to(device)

        return PreloadedDataset(images=data, targets=targets, transform=transform, device=device)

    def from_folder(self, main_dir, transform=None, use_tqdm=True, device='cpu'):

        to_tensor = transforms.ToTensor()
        class_dirs = os.listdir(main_dir)
        loop = tqdm(enumerate(class_dirs), total=len(class_dirs), leave=False) if use_tqdm else enumerate(class_dirs)
        images = []
        targets = []

        for class_idx, class_name in loop:
            class_dir = os.path.join(main_dir, class_name)
            image_names = os.listdir(class_dir)
            class_images = []
            for file_name in image_names:
                img_loc = os.path.join(class_dir, file_name)
                class_images.append(to_tensor(Image.open(img_loc).convert("RGB")))

            class_images = torch.stack(class_images).to(self.device)
            class_targets = (torch.ones(len(class_images)) * class_idx).type(torch.LongTensor).to(self.device)

            images.append(class_images)
            targets.append(class_targets)
        
        images = torch.cat(images)
        targets = torch.cat(targets)

        return PreloadedDataset(images=images, targets=targets, transform=transform, device=device)
            
    #  Transforms the data in batches so as not to overload memory
    def update_transformed_images(self, transform_device=torch.device('cuda'), batch_size=512):
        if self.transform is None:
            self.transformed_images = self.images
            return

        if transform_device is None:
            transform_device = self.device
        
        transformed_images = []
        targets = []
        low = 0
        high = batch_size
        while low < len(self.images):
            if high > len(self.images):
                high = len(self.images)
            out = self.transform(self.images[low:high].to(transform_device))
            if type(out) == tuple:
                transformed_images.append(out[0].to(self.device).detach())
                targets.append(out[1].to(self.device).detach())
            else:
                transformed_images.append(out.to(self.device).detach())
            low += batch_size
            high += batch_size
        
        self.transformed_images = torch.cat(transformed_images)
        if len(targets) > 0:
            self.targets = torch.cat(targets)
        
    #  Now a man who needs no introduction
    def __len__(self):
        return len(self.images)
    
    #  Returns images which have already been transformed - unless self.transform is none
    #  This saves us from transforming individual images, which is very slow.
    def __getitem__(self, idx):
        return self.transformed_images[idx], self.targets[idx]        
    
    def _shuffle(self):
        indices = torch.randperm(self.images.shape[0])
        self.images = self.images[indices]
        self.targets = self.targets[indices]
        self.transformed_images = self.transformed_images[indices]
        if not self.shuffled:
            self.shuffled = True  
    
    def to_dtype(self, dtype):
        self.images = self.images.to(dtype)
        self.transformed_images = self.transformed_images.to(dtype)
        return self
    
    def to(self, to):
        if type(to) == str:
            if to == 'cpu':
                to = torch.device('cpu')
            elif to == 'cuda':
                to = torch.device('cuda')
            elif to == 'mps':
                to = torch.device('mps')
            elif to == 'float32':
                to = torch.float32
            elif to == 'float16':
                to = torch.float16
            elif to == 'int32':
                to = torch.int32
            elif to == 'int16':
                to = torch.int16

        if type(to) == torch.device:
            self.images = self.images.to(to)
            self.targets = self.targets.to(to)
            self.transformed_images = self.transformed_images.to(to)
            self.device = to
        elif type(to) == torch.dtype:
            self.images = self.images.to(dtype=to)
            self.transformed_images = self.transformed_images.to(dtype=to)
            self.dtype = to
        else:
            raise ValueError(f"Invalid type for to: {type(to)}")

        return self

    def get_balanced_subset(self, n_train: int, shuffle=True, transform=None, device=None):
        n_per_class = n_train // len(self.classes)
        train_data, train_labels, test_data, test_labels = [], [], [], []
        for i in range(len(self.classes)):
            indices = torch.where(self.targets == i)[0]
            if shuffle:
                shuffle_idx = torch.randperm(len(indices))
                indices = indices[shuffle_idx]
            train_data.append(self.images[indices[:n_per_class]])
            train_labels.append(self.targets[indices[:n_per_class]])
            test_data.append(self.images[indices[n_per_class:]])
            test_labels.append(self.targets[indices[n_per_class:]])

        train_data = torch.cat(train_data)
        train_labels = torch.cat(train_labels)
        test_data = torch.cat(test_data)
        test_labels = torch.cat(test_labels)

        if transform is None:
            transform = self.transform
        if device is None:
            device = self.device
        train = PreloadedDataset(images=train_data, targets=train_labels, transform=transform, device=device)
        test = PreloadedDataset(images=test_data, targets=test_labels, transform=transform, device=device)

        return train, test