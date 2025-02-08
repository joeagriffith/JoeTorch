import os
from torch.utils.tensorboard import SummaryWriter
import time

def get_writer(out_dir: str, experiment_name: str, trial_name: str, **kwargs):
    trial_log_dir = out_dir + f'{experiment_name}/logs/raw/{trial_name}'
    agg_log_dir = out_dir + f'{experiment_name}/logs/agg/{trial_name}'

    # remove aggregation as it needs to be recalculated with this run.
    if os.path.exists(agg_log_dir):
        for f in os.listdir(agg_log_dir):
            os.remove(agg_log_dir + '/' + f)

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    return SummaryWriter(trial_log_dir + f'/{time_str}')