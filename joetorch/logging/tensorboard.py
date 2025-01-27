import os
from torch.utils.tensorboard import SummaryWriter
import time

def get_writer(log_dir: str, experiment_name: str, trial_name: str):
    trial_log_dir = log_dir + f'raw/{experiment_name}/{trial_name}'
    agg_log_dir = log_dir + f'agg/{experiment_name}/{trial_name}'

    # remove aggregation as it needs to be recalculated with this run.
    if os.path.exists(agg_log_dir):
        for f in os.listdir(agg_log_dir):
            os.remove(agg_log_dir + '/' + f)

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    return SummaryWriter(trial_log_dir + f'/{time_str}')