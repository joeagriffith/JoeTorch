import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
import xml.etree.ElementTree as ET

from joetorch.datasets.dataset import PreloadedDataset

class ImageNet(Dataset):
    def __init__(self, root, split, subset_size=None, size=(224, 224)):
        self.root = root
        self.split = split
        assert split in ['train', 'val', 'test']

        self.data_dir = os.path.join(root, f'ILSVRC/Data/CLS-LOC/{split}')
        self.to_tensor = transforms.ToTensor()
        self.size = size

        self.folder_to_label = {folder: i for i, folder in enumerate(os.listdir(os.path.join(root, 'ILSVRC/Data/CLS-LOC/train')))}

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),  # Random crop & resize
                transforms.RandomHorizontalFlip(),  # Augmentation
                transforms.ToTensor(),              # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
            ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.image_paths = []
        self.labels = []
        if split == 'train':
            subdirs = [os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]

            for d in subdirs:
                class_name = d.split('/')[-1]
                for fname in os.listdir(d):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(d, fname))
                        self.labels.append(self.folder_to_label[class_name])
        else:
            label_dir = os.path.join(root, f'ILSVRC/Annotations/CLS-LOC/{split}')
            for fname in os.listdir(self.data_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(self.data_dir, fname))
                    annotation = os.path.join(label_dir, fname.replace('.JPEG', '.xml'))
                    tree = ET.parse(annotation)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        self.labels.append(self.folder_to_label[label])

        if subset_size is not None:
            indices = np.random.choice(len(self.image_paths), subset_size, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = self.labels[idx]

        return image, label