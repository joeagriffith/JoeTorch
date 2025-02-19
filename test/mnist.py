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
import wandb

# Training Hyperparameters
cfg = {
    'experiment_name': 'mnist',
    'out_dir': 'test/out/',
    'root': '../../datasets/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': 'bfloat16',
    'val_ratio': 0.1,
    'num_epochs': 50,
    'batch_size': 256,
    'start_lr': 1e-3,
    'end_lr': 1e-4,
    'lr_warmup_epochs': 10,
    'start_wd': 4e-3,
    'end_wd': 4e-2,
    'out_dim': 20,
}

epoch_hyperparams = {
    'lr': cosine_schedule(base=cfg['start_lr'], end=cfg['end_lr'], T=cfg['num_epochs'], warmup=cfg['lr_warmup_epochs']),
    'wd': cosine_schedule(base=cfg['start_wd'], end=cfg['end_wd'], T=cfg['num_epochs']),
}
dtype = getattr(torch, cfg['dtype'])
train_dataset = MNIST(root=cfg['root'], split='train', val_ratio=cfg['val_ratio'], augment=True, dtype=dtype, device=cfg['device'], download=False)
val_dataset = MNIST(root=cfg['root'], split='val', val_ratio=cfg['val_ratio'], dtype=dtype, device=cfg['device'], download=False)
test_dataset = MNIST(root=cfg['root'], split='test', dtype=dtype, device=cfg['device'], download=False)

# MLP trial
cfg['trial_name'] = 'mlp_ae_mseloss'
cfg['mode'] = 'mlp'
model = MNIST_AE(out_dim=cfg['out_dim'], mode=cfg['mode']).to(cfg['device'])
optimiser = get_optimiser(model, optim='AdamW')
wandb.init(entity='joeagriffith-home', project='joetorch', name=cfg['trial_name'], config=cfg, dir=cfg['out_dir'])
wandb.log({'model': str(model).replace('\n', '<br/>').replace(' ', '&nbsp;')})
save_dir = cfg['out_dir'] + f'{cfg['experiment_name']}/models/{cfg['trial_name']}'
train(
    model=model,
    train_dataset=train_dataset,
    optimiser=optimiser,
    val_dataset=val_dataset,
    epoch_hyperparams=epoch_hyperparams,
    save_dir=save_dir,
    **cfg,
)
wandb.finish()

# CNN trial
cfg['trial_name'] = 'cnn_ae_mseloss'
cfg['mode'] = 'cnn'
model = MNIST_AE(out_dim=cfg['out_dim'], mode=cfg['mode']).to(cfg['device'])
optimiser = get_optimiser(model, optim='AdamW')
wandb.init(entity='joeagriffith-home', project='joetorch', name=cfg['trial_name'], config=epoch_hyperparams, dir=cfg['out_dir'])
wandb.log({'model': str(model).replace('\n', '<br/>').replace(' ', '&nbsp;')})
save_dir = cfg['out_dir'] + f'{cfg['experiment_name']}/models/{cfg['trial_name']}'
train(
    model=model,
    train_dataset=train_dataset,
    optimiser=optimiser,
    val_dataset=val_dataset,
    epoch_hyperparams=epoch_hyperparams,
    save_dir=save_dir, **cfg)
wandb.finish()
