import os
import shutil
import torch

import sys
sys.path.append('../') # if run from inside scripts
sys.path.append('') # if run from root

print(sys.path)

from joetorch.datasets import MNIST
from joetorch.nn import *
from joetorch.optim import *
from joetorch.logging import get_writer

# Training Hyperparameters
experiment_name = 'mnist'
out_dir = 'test/out/'
num_epochs = 50
batch_size = 256
start_lr, end_lr = 1e-3, 1e-4
lr_warmup_epochs = 10
start_wd, end_wd = 4e-3, 4e-2
epoch_hyperparams = {
    'lr': cosine_schedule(base=start_lr, end=end_lr, T=num_epochs, warmup=lr_warmup_epochs),
    'wd': cosine_schedule(base=start_wd, end=end_wd, T=num_epochs),
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

root = '../../datasets/'
val_ratio = 0.1
train_dataset = MNIST(root=root, split='train', val_ratio=val_ratio, augment=True, dtype=dtype, device=device, download=False)
val_dataset = MNIST(root=root, split='val', val_ratio=val_ratio, dtype=dtype, device=device, download=False)
test_dataset = MNIST(root=root, split='test', dtype=dtype, device=device, download=False)

# MLP trial
trial_name = 'mlp_ae_mseloss'
model = MNIST_AE(out_dim=20, mode='mlp').to(device)
optimiser = get_optimiser(model, optim='AdamW')
writer = get_writer(out_dir, experiment_name, trial_name)
save_dir = out_dir + f'{experiment_name}/models/{trial_name}'
train(model, train_dataset, optimiser, num_epochs, batch_size, val_dataset, writer, compute_dtype=dtype, epoch_hyperparams=epoch_hyperparams, save_dir=save_dir)

# CNN trial
trial_name = 'cnn_ae_mseloss'
model = MNIST_AE(out_dim=20, mode='cnn').to(device)
optimiser = get_optimiser(model, optim='AdamW')
writer = get_writer(out_dir, experiment_name, trial_name)
save_dir = out_dir + f'{experiment_name}/models/{trial_name}'
train(model, train_dataset, optimiser, num_epochs, batch_size, val_dataset, writer, compute_dtype=dtype, epoch_hyperparams=epoch_hyperparams, save_dir=save_dir)