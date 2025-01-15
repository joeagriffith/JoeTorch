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
    def __init__(self, main_dir, shape, transform=None, shuffle=False, use_tqdm=True):
        self.main_dir = main_dir
        self.shape = shape
        self.transform = transform
        self.classes = os.listdir(main_dir)
        self.shuffled = shuffle #  This flag is useful for cross_val_split_by_class()  
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')
            
        self.images = torch.zeros(shape).to(self.device)
        self.targets = torch.zeros(0).type(torch.LongTensor).to(self.device)
        
        pre_transform = transforms.ToTensor()
        self.transform = remove_to_tensor(transform)                
        
        #  preload images
        if self.main_dir is not None:
            loop = tqdm(enumerate(self.classes), total=len(self.classes), leave=False) if use_tqdm else enumerate(self.classes)
            for class_idx, class_name in loop:
                class_dir = os.path.join(self.main_dir, class_name)
                image_names = os.listdir(class_dir)
                class_images = []
                for file_name in image_names:
                    img_loc = os.path.join(class_dir, file_name)
                    class_images.append(pre_transform(Image.open(img_loc).convert("RGB")))

                class_images = torch.stack(class_images).to(self.device)
                class_targets = (torch.ones(len(class_images)) * class_idx).type(torch.LongTensor).to(self.device)

                self.images = torch.cat([self.images, class_images])
                self.targets = torch.cat([self.targets, class_targets])
            
            #  Transformed_images stores a transformed copy of images.
            #  This enables us to keep a virgin copy of the original images
            #  which we can use at each epoch to generate different transformed images
            #  Note: we must remember to call dataset.transform() when requesting transformed images
        if self.transform is None:
            self.transformed_images = self.images
        else:
            self.transformed_images = self.transform(self.images)
            
        if shuffle:
            self._shuffle()
        
    #  Useful for loading data which is stored in a different format to TinyImageNet30
    def from_dataset(dataset, transform, device="cpu", use_tqdm=True):
        preloaded_dataset = PreloadedDataset(None, dataset.__getitem__(0)[0].shape, use_tqdm=use_tqdm)
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
        preloaded_dataset.device = device
        preloaded_dataset.transform = transform
        preloaded_dataset.images = torch.stack(data).to(device)
        preloaded_dataset.targets = torch.stack(targets).to(device)
        if transform is not None:
            preloaded_dataset.transformed_images = transform(torch.stack(data).to(device))
        else:
            preloaded_dataset.transformed_images = torch.stack(data).to(device)
        
        return preloaded_dataset
            
    #  Transforms the data in batches so as not to overload memory
    def apply_transform(self, device=torch.device('cuda'), batch_size=500):
        if self.transform is not None:
            if device is None:
                device = self.device
            
            low = 0
            high = batch_size
            while low < len(self.images):
                if high > len(self.images):
                    high = len(self.images)
                self.transformed_images[low:high] = self.transform(self.images[low:high].to(device)).to(self.device)
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
        assert self.images.device == self.targets.device == self.transformed_images.device, "All images, targets, and transformed images must be on the same device"
        return self.images.device
    
    @property
    def dtype(self):
        assert self.images.dtype == self.transformed_images.dtype, "All images, targets, and transformed images must be on the same dtype"
        return self.images.dtype, self.targets.dtype
    
