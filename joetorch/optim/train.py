import torch
from joetorch.datasets.dataset import PreloadedDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, List, Dict
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def train(
        model: torch.nn.Module,
        train_dataset: PreloadedDataset,
        val_dataset: PreloadedDataset,
        optimiser: torch.optim.Optimizer,
        num_epochs: int,
        batch_size: int,
        writer: Optional[SummaryWriter] = None,
        compute_dtype: Optional[torch.dtype] = None,
        epoch_hyperparams: Dict[str, torch.Tensor] = {},
):
    """
    Train a model for a given number of epochs.

    Args:
        model: The model to train.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        optimiser: The optimiser to use.
        num_epochs: The number of epochs to train for.
        batch_size: The batch size.
        writer: The writer to use for logging.
        mixed_precision: Whether to use mixed precision.
        epoch_hyperparams: The hyperparameters to use for each epoch.
    """
    compute_device = next(model.parameters()).device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loop = tqdm(range(num_epochs), desc='Epochs', leave=False)
    for epoch in loop:

        # ============================ DATA AUGMENTATION ============================

        if hasattr(train_dataset, 'augment') and train_dataset.augment:   
            train_dataset.update_transformed_images()
        if hasattr(val_dataset, 'augment') and val_dataset.augment:
            val_dataset.update_transformed_images()
        
        # ========================== HYPERPARAMETER UPDATE ==========================

        epoch_loss_args = {}
        for key, value in epoch_hyperparams.items():
            if key == 'lr':
                for param_group in optimiser.param_groups:
                    param_group['lr'] = value[epoch]
            elif key == 'wd':
                for param_group in optimiser.param_groups:
                    if param_group['weight_decay'] != 0.0:
                        param_group['weight_decay'] = value[epoch]
            else:
                epoch_loss_args[key] = value[epoch]

        # ============================ TRAINING ============================

        model.train()
        epoch_train_metrics = {}
        for batch in train_loader:
            optimiser.zero_grad()
            with torch.autocast(device_type=compute_device.type, dtype=compute_dtype, enabled=compute_dtype is not None):
                train_metrics = model.loss(batch, **epoch_loss_args)
                train_metrics['loss'].backward()
                optimiser.step()
            
            for key, value in train_metrics.items():
                if key not in epoch_train_metrics:
                    epoch_train_metrics[key] = []
                epoch_train_metrics[key].append(value.detach().cpu().item())

        # ============================ VALIDATION ============================

        model.eval()
        epoch_val_metrics = {}
        for batch in val_loader:
            with torch.no_grad():
                with torch.autocast(device_type=compute_device.type, dtype=compute_dtype, enabled=compute_dtype is not None):
                    val_metrics = model.loss(batch, **epoch_loss_args)

            for key, value in val_metrics.items():
                if key not in epoch_val_metrics:
                    epoch_val_metrics[key] = []
                epoch_val_metrics[key].append(value.detach().cpu().item())
        
        # ============================ LOGGING ============================

        # average metrics over the epoch
        for key, value in epoch_train_metrics.items():
            epoch_train_metrics[key] = np.mean(value)
        for key, value in epoch_val_metrics.items():
            epoch_val_metrics[key] = np.mean(value)

        if writer is not None:
            for key, value in epoch_train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch)
            for key, value in epoch_val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)

        loop.set_postfix({key: round(value, 3) for key, value in epoch_train_metrics.items()})
