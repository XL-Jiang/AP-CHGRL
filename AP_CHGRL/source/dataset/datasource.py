import torch
from omegaconf import DictConfig, open_dict
import h5py
import numpy as np


def load_cpac_abide1_data(cfg: DictConfig):
    try:
        with h5py.File(cfg.dataset.path, 'r') as f:
            print("OK:", list(f.keys()))
    except Exception as e:
        print("Fail:", e)

    with h5py.File(cfg.dataset.path, 'r') as f:
        final_wsr1 = f["AAL-116_wsr"][:]  # ndarray
        final_wsr2 = f["BNA-246_wsr"][:]  # ndarray
        final_pearson1 = f["AAL-116_corr"][:]  # ndarray
        final_pearson2 = f["BNA-246_corr"][:]  # ndarray
        labels = f["label"][:]
        IAMP_edge = f["JC_adj"][:]
        site = np.array([s.decode('utf-8') for s in f["site"]], dtype='<U8')

    final_pearson1, final_pearson2, final_wsr1, final_wsr2, IAMP_edge, labels = [torch.from_numpy(
        data).float() for data in (final_pearson1, final_pearson2, final_wsr1, final_wsr2, IAMP_edge, labels)]

    with (open_dict(cfg)):

        cfg.dataset.node_sz = [final_pearson1.shape[1], final_pearson2.shape[1]]  # [node_size1, node_size2]
        cfg.dataset.node_dims_dict = {
            'atlas0': final_pearson1.shape[2],
            'atlas1': final_pearson2.shape[2]
        }

    return final_pearson1, final_pearson2,final_wsr1, final_wsr2, IAMP_edge, labels, site



def load_my_adni2_data(cfg: DictConfig):
    try:
        with h5py.File(cfg.dataset.path, 'r') as f:
            print("OK:", list(f.keys()))
    except Exception as e:
        print("Fail:", e)

    with h5py.File(cfg.dataset.path, 'r') as f:

        final_wsr1 = f["AAL-90_wsr"][:]  # ndarray
        final_wsr2 = f["Schaefer_7-100_wsr"][:]  # ndarray

        final_pearson1 = f["AAL-90_corr"][:]  # ndarray
        final_pearson2 = f["Schaefer_7-100_corr"][:]  # ndarray

        labels = f["label"][:]
        IAMP_edge = f["JC_adj"][:]
        site = np.array([s.decode('utf-8') for s in f["site"]], dtype='<U8')

    final_pearson1, final_pearson2, final_wsr1, final_wsr2, IAMP_edge, labels = [torch.from_numpy(
        data).float() for data in (final_pearson1, final_pearson2, final_wsr1, final_wsr2, IAMP_edge, labels)]

    with (open_dict(cfg)):

        cfg.dataset.node_sz = [final_pearson1.shape[1], final_pearson2.shape[1]]  # [node_size1, node_size2]
        cfg.dataset.node_dims_dict = {
            'atlas0': final_pearson1.shape[2],
            'atlas1': final_pearson2.shape[2]
        }

    return final_pearson1, final_pearson2,final_wsr1, final_wsr2,  IAMP_edge, labels, site


