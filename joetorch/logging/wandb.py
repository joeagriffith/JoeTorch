import wandb
from typing import Dict

def wandb_init(entity: str, project: str, name: str, config: Dict[str, any]):
    wandb.init(entity=entity, project=project, name=name, config=config)
    return wandb.log