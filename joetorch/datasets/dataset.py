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
    def __init__(self, main_dir, shape, transform=None, shuffle=False, use_tqdm=True, augment=False, device='cpu'):
        self.main_dir = main_dir
        self.shape = shape
        self.transform = transform
        self.augment = augment
        self.classes = os.listdir(main_dir)
        self.shuffled = shuffle #  This flag is useful for cross_val_split_by_class()  
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')
            
        self.images = None
        self.targets = None
        self.transformed_images = None

        pre_transform = transforms.ToTensor()
        self.transform = remove_to_tensor(transform)                
        
        #  preload images
        if self.main_dir is not None:
            loop = tqdm(enumerate(self.classes), total=len(self.classes), leave=False) if use_tqdm else enumerate(self.classes)
            images = []
            targets = []
            for class_idx, class_name in loop:
                class_dir = os.path.join(self.main_dir, class_name)
                image_names = os.listdir(class_dir)
                class_images = []
                for file_name in image_names:
                    img_loc = os.path.join(class_dir, file_name)
                    class_images.append(pre_transform(Image.open(img_loc).convert("RGB")))

                class_images = torch.stack(class_images).to(self.device)
                class_targets = (torch.ones(len(class_images)) * class_idx).type(torch.LongTensor).to(self.device)

                images.append(class_images)
                targets.append(class_targets)
            
            self.images = torch.cat(images)
            self.targets = torch.cat(targets)
            self.update_transformed_images()
            
        if shuffle:
            self._shuffle()
        
    #  Useful for loading data which is stored in a different format to TinyImageNet30
    def from_dataset(dataset, transform, device="cpu", use_tqdm=True, augment=False):
        preloaded_dataset = PreloadedDataset(None, dataset.__getitem__(0)[0].shape, use_tqdm=use_tqdm, augment=augment)
        data = []
        targets = []
        loop = tqdm(range(len(dataset)), leave=False) if use_tqdm else range(len(dataset))
        for i in loop:
            d, t = dataset.__getitem__(i)
            if type(t) is not torch.Tensor:
                t = torch.tensor(t)
            data.append(d)
            targets.append(t)
            
        assert type(data[0]) == torch.Tensor, print(f"Data is {type(data[0])} not torch.Tensor")
        assert type(targets[0]) == torch.Tensor, print(f"Targets is {type(targets[0])} not torch.Tensor")
        transform = remove_to_tensor(transform)
        
        preloaded_dataset.shape = data[0].shape
        preloaded_dataset.transform = transform
        preloaded_dataset.images = torch.stack(data).to(device)
        preloaded_dataset.targets = torch.stack(targets).to(device)
        preloaded_dataset.update_transformed_images()
        
        return preloaded_dataset

    def from_tensors(data, targets, transform, device="cpu", use_tqdm=True, augment=False):
        assert type(data) == torch.Tensor, "Data must be a torch.Tensor"
        assert type(targets) == torch.Tensor, "Targets must be a torch.Tensor"

        if data.device != device:
            data = data.to(device)
        if targets.device != device:
            targets = targets.to(device)

        preloaded_dataset = PreloadedDataset(None, data[0].shape, use_tqdm=use_tqdm, augment=augment)
        preloaded_dataset.images = data
        preloaded_dataset.targets = targets
        preloaded_dataset.shape = data[0].shape
        preloaded_dataset.device = device
        preloaded_dataset.transform = transform
        preloaded_dataset.update_transformed_images()
        
        return preloaded_dataset
            
    #  Transforms the data in batches so as not to overload memory
    def update_transformed_images(self, transform_device=torch.device('cuda'), batch_size=500):
        if self.transform is None:
            self.transformed_images = self.images
            return

        if self.transformed_images is None:
            self.transformed_images = torch.zeros_like(self.images)
        elif self.transformed_images.device != self.images.device:
            self.transformed_images = self.transformed_images.to(self.images.device)
        elif self.transformed_images.dtype != self.images.dtype:
            self.transformed_images = self.transformed_images.to(self.images.dtype)

        if transform_device is None:
            transform_device = self.device
        
        low = 0
        high = batch_size
        while low < len(self.images):
            if high > len(self.images):
                high = len(self.images)

            self.transformed_images[low:high] = self.transform(self.images[low:high].to(transform_device)).to(self.device).detach()
            low += batch_size
            high += batch_size
        
        
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
        self.targets = self.targets.to(dtype)
        self.transformed_images = self.transformed_images.to(dtype)
        return self
    
    def to(self, device):
        self.images = self.images.to(device)
        self.targets = self.targets.to(device)
        self.transformed_images = self.transformed_images.to(device)
        return self
    
    @property
    def device(self):
        assert self.images.device == self.targets.device, f"Images and targets must be on the same device, {self.images.device} != {self.targets.device}"
        assert self.images.device == self.transformed_images.device, f"Images and transformed images must be on the same device, {self.images.device} != {self.transformed_images.device}"
        return self.images.device
    
    @property
    def dtype(self):
        assert self.images.dtype == self.transformed_images.dtype, "All images, targets, and transformed images must be on the same dtype"
        return self.images.dtype, self.targets.dtype
    

def get_balanced_subset(data, labels, n_train: int, shuffle=True):
    n_per_class = n_train // 10
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for i in range(10):
        indices = torch.where(labels == i)[0]
        if shuffle:
            shuffle_idx = torch.randperm(len(indices))
            indices = indices[shuffle_idx]
        train_data.append(data[indices[:n_per_class]])
        train_labels.append(labels[indices[:n_per_class]])
        test_data.append(data[indices[n_per_class:]])
        test_labels.append(labels[indices[n_per_class:]])

    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)
    test_data = torch.cat(test_data)
    test_labels = torch.cat(test_labels)

    return train_data, train_labels, test_data, test_labels
