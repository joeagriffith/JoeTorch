import os
import torch
from joetorch.datasets.dataset import PreloadedDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, List, Dict, Tuple
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def train(
        model: torch.nn.Module,
        train_dataset: PreloadedDataset,
        optimiser: torch.optim.Optimizer,
        num_epochs: int,
        batch_size: int,
        val_dataset: Optional[PreloadedDataset] = None,
        writer: Optional[SummaryWriter] = None,
        compute_dtype: str = 'float32',
        compile: bool = False,
        epoch_hyperparams: Dict[str, any] = {},
        loss_args: Dict[str, any] = {},
        save_dir: Optional[str] = None,
        save_metric: Tuple[str, str, str] = ('minimise', 'val', 'loss'),
        save_every: Optional[int] = None,
        save_last: bool = False,
        target_networks: Optional[List[Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]]] = None,
        **kwargs,
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
        loss_args: Extra arguments to pass to the loss function.
        save_dir: The directory to save the model.
        save_metric: Tuple of ('minimise'/'maximise', 'train'/'val', {metric}).
    """
    assert save_metric[0] in ['minimise', 'maximise']
    assert save_metric[1] in ['train', 'val']

    if compile:
        loss_fn = torch.compile(model.loss)
    else:
        loss_fn = model.loss

    compute_dtype = getattr(torch, str(compute_dtype).split('.')[-1])
    compute_device = next(model.parameters()).device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loop = tqdm(range(num_epochs), desc='Epochs', leave=False)
    best_metric = torch.inf if save_metric[0] == 'minimise' else -torch.inf
    for epoch in loop:

        # ========================== HYPERPARAMETER UPDATE ==========================

        for key, value in epoch_hyperparams.items():
            if key == 'lr':
                for param_group in optimiser.param_groups:
                    param_group['lr'] = value[epoch]
            elif key == 'wd':
                for param_group in optimiser.param_groups:
                    if param_group['weight_decay'] != 0.0:
                        param_group['weight_decay'] = value[epoch]
            
        epoch_loss_args = {}
        for key, value in loss_args.items():
            if (isinstance(value, torch.Tensor) or isinstance(value, np.ndarray) or isinstance(value, list)) and len(value) == num_epochs:
                epoch_loss_args[key] = value[epoch]
            else:
                epoch_loss_args[key] = value
                    
        # ============================ TRAINING ============================
        if train_dataset.transform is not None:
            train_dataset.update_transformed_images()

        model.train()
        epoch_train_metrics = {}
        for batch in train_loader:
            optimiser.zero_grad()
            with torch.autocast(device_type=compute_device.type, dtype=compute_dtype, enabled=compute_dtype is not None):
                if 'train_metrics' in locals():
                    del train_metrics
                    torch.cuda.empty_cache()
                train_metrics = loss_fn(batch, **epoch_loss_args)
                train_metrics['loss'].backward()
                optimiser.step()
            
            for key, value in train_metrics.items():
                if key not in epoch_train_metrics:
                    epoch_train_metrics[key] = []
                epoch_train_metrics[key].append(value.detach().cpu().item())

        for key, value in epoch_train_metrics.items():
            epoch_train_metrics[key] = np.mean(value)

        if writer is not None:
            for key, value in epoch_train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch)
            
        postfix = {key: round(value.item(), 3) for key, value in epoch_train_metrics.items()}
        # ============================ VALIDATION ============================

        if val_loader is not None:
            if val_dataset.transform is not None:
                val_dataset.update_transformed_images()

            model.eval()
            epoch_val_metrics = {}
            for batch in val_loader:
                with torch.no_grad():
                    with torch.autocast(device_type=compute_device.type, dtype=compute_dtype, enabled=compute_dtype is not None):
                        val_metrics = loss_fn(batch, **epoch_loss_args)

                for key, value in val_metrics.items():
                    if key not in epoch_val_metrics:
                        epoch_val_metrics[key] = []
                    epoch_val_metrics[key].append(value.detach().cpu().item())
            
            # ============================ LOGGING ============================

            # average metrics over the epoch
            for key, value in epoch_val_metrics.items():
                epoch_val_metrics[key] = np.mean(value)

            if writer is not None:
                for key, value in epoch_val_metrics.items():
                    writer.add_scalar(f'val/{key}', value, epoch)
            
            for val_key, val_value in epoch_val_metrics.items():
                val_value = round(val_value.item(), 3)
                if val_key in postfix:
                    train_value = postfix[val_key]
                    postfix[val_key] = (train_value, val_value)
                else:
                    postfix[val_key] = val_value

        loop.set_postfix(postfix)

        # ============================ TARGET NETWORK UPDATE ============================

        if target_networks is not None:
            for target_network, network, taus in target_networks:
                for param, target_param in zip(network.parameters(), target_network.parameters()):
                    target_param.data.copy_(taus[epoch] * param.data + (1 - taus[epoch]) * target_param.data)

        # ============================ SAVING ============================

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            metrics = epoch_val_metrics if save_metric[1] == 'val' else epoch_train_metrics
            if (save_metric[0] == 'minimise' and metrics[save_metric[2]] < best_metric) or (save_metric[0] == 'maximise' and metrics[save_metric[2]] > best_metric):
                best_metric = metrics[save_metric[2]]
                for f in os.listdir(save_dir):
                    if 'best' in f:
                        os.remove(os.path.join(save_dir, f))
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_{save_metric[1]}_{save_metric[2]}_{epoch}.pth'))
            if save_every is not None and epoch % save_every == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_{epoch}.pth'))
            elif save_last:
                for f in os.listdir(save_dir):
                    if 'recent' in f:
                        os.remove(os.path.join(save_dir, f))
                torch.save(model.state_dict(), os.path.join(save_dir, f'recent_{epoch}.pth'))