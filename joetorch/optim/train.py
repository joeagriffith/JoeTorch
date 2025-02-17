import os
import torch
from joetorch.datasets.dataset import PreloadedDataset
from typing import Optional, Union, List, Dict, Tuple
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Callable


def train(
        model: torch.nn.Module,
        train_dataset: PreloadedDataset,
        optimiser: torch.optim.Optimizer,
        num_epochs: int,
        batch_size: int,
        val_dataset: Optional[PreloadedDataset] = None,
        compute_dtype: str = 'float32',
        compile: bool = False,
        epoch_hyperparams: Dict[str, any] = {},
        loss_args: Dict[str, any] = {},
        save_dir: Optional[str] = None,
        save_metric: Tuple[str, str, str] = ('minimise', 'val', 'loss'),
        save_every: Optional[int] = None,
        save_last: bool = False,
        target_networks: Optional[List[Tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]]] = None,
        max_grad_norm: Optional[float] = None,
        grad_norm_depth: int = 0,
        callback: Optional[Callable] = None,
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
        save_every: The number of epochs between saving the model.
        save_last: Whether to save the model after the last epoch.
        target_networks: List of tuples of (target_network, network, taus).
        max_grad_norm: The maximum gradient norm.
        grad_norm_depth: The depth of the gradient norm to compute.
        wandb_init: The initialisation arguments for wandb.
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

                if max_grad_norm is not None:
                    if grad_norm_depth == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        train_metrics[f'{model.__class__.__name__}_grad_norm'] = grad_norm
                    else:
                        def get_submodules(module, depth, prefix=''):
                            if depth == 0:
                                return [(prefix.rstrip('.'), module)]
                            
                            submods = []
                            for name, submod in module.named_children():
                                submod_name = f"{prefix}{name}."
                                submods.extend(get_submodules(submod, depth-1, submod_name))
                            return submods if submods else [(prefix.rstrip('.'), module)]

                        for name, submod in get_submodules(model, grad_norm_depth):
                            if any(p.grad is not None for p in submod.parameters()):
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    submod.parameters(), 
                                    max_grad_norm
                                )
                                train_metrics[f'{name}_grad_norm'] = grad_norm

                optimiser.step()

            if callback is not None:
                callback('train_batch_end', locals())
            
            for key, value in train_metrics.items():
                if key not in epoch_train_metrics:
                    epoch_train_metrics[key] = []
                epoch_train_metrics[key].append(value.detach().cpu().item())

        for key, value in epoch_train_metrics.items():
            epoch_train_metrics[key] = np.mean(value)

        if wandb.run is not None:
            for key, value in epoch_train_metrics.items():
                wandb.log({f'train/{key}': value}, step=epoch+1)
            
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

                if callback is not None:
                    callback('val_batch_end', locals())

                for key, value in val_metrics.items():
                    if key not in epoch_val_metrics:
                        epoch_val_metrics[key] = []
                    epoch_val_metrics[key].append(value.detach().cpu().item())
            
            # ============================ LOGGING ============================

            # average metrics over the epoch
            for key, value in epoch_val_metrics.items():
                epoch_val_metrics[key] = np.mean(value)

            if wandb.run is not None:
                for key, value in epoch_val_metrics.items():
                    wandb.log({f'val/{key}': value}, step=epoch+1)
            
            for val_key, val_value in epoch_val_metrics.items():
                val_value = round(val_value.item(), 3)
                if val_key in postfix:
                    train_value = postfix[val_key]
                    postfix[val_key] = (train_value, val_value)
                else:
                    postfix[val_key] = val_value


        loop.set_postfix(postfix)

        if callback is not None:
            callback('epoch_end', locals())

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