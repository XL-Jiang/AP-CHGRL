from omegaconf import DictConfig, open_dict
from .datasource import load_cpac_abide1_data,load_my_adni2_data
from .dataloader import init_kfold_heterograph_dataloaders_no_val
from typing import List
import torch.utils as utils

def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:
    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)
    dataloaders = init_kfold_heterograph_dataloaders_no_val(cfg, *datasets, n_splits=cfg.kfold.n_splits)

    return dataloaders
