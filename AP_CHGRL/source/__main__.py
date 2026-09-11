import itertools
import json
from datetime import datetime
import wandb
import hydra
from pathlib import Path
from omegaconf import DictConfig, open_dict, OmegaConf
from dataset import dataset_factory
from models import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import os
import time
import numpy as np
import random
import torch


def get_timestamp():
    '''
        返回当前时间的时间戳
    '''
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_RUN_ID") and k not in exclude:
            print(k)
            print(v)


def model_training(cfg: DictConfig, SEED,hyperparams=None):
    if hyperparams:
        with open_dict(cfg):
            for key, value in hyperparams.items():
                if '.' in key:
                    parts = key.split('.')
                    current = cfg
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], value)
                else:
                    setattr(cfg, key, value)



    dataloaders = dataset_factory(cfg)


    if isinstance(dataloaders, list) and len(dataloaders) > 0 and isinstance(dataloaders[0], tuple):
        acc_list, auc_list, sen_list, spec_list = [], [], [], []
        for fold_idx, (train_loader, test_loader) in enumerate(dataloaders):
            with open_dict(cfg):
                cfg.unique_id = f"lr{cfg.optimizer.base_lr}_wd{cfg.optimizer.weight_decay}_wsr{cfg.dataset.wsr}_sc{cfg.model.sc_loss_lamda}_pc{cfg.model.pc_loss_lamda}_fold{fold_idx}"
            run = wandb.init(
                project=cfg.project,
                group=f"{cfg.group_name}",
                name=f"{cfg.group_name}_{cfg.unique_id} ",
                reinit=True,
                settings=wandb.Settings(init_timeout=120)
                # mode = "offline"
            )
            print(f"====================fold{fold_idx}====================")

            logger = logger_factory(cfg)
            model = model_factory(cfg)
            if cfg.state == 'testing':
                print("############testing##############")
                cfg.training.epochs = 1
                save_dir = Path(cfg.log_path) / cfg.unique_id
                model_weight_path = save_dir / "model.pt"
                if not os.path.exists(model_weight_path):
                    raise FileNotFoundError(f"模型权重文件不存在：{model_weight_path}")
                state_dict = torch.load(model_weight_path)
                model.load_state_dict(state_dict)
                logger.info(f"成功加载模型权重：{model_weight_path}")
            optimizers = optimizers_factory(model, cfg.optimizer)
            lr_schedulers = lr_scheduler_factory(cfg.optimizer, cfg)

            training = training_factory(cfg, model, optimizers, lr_schedulers,
                                        [train_loader, None, test_loader], logger)
            t_acc, t_auc, t_sen, t_spec = training.train()
            acc_list.append(t_acc)
            auc_list.append(t_auc)
            sen_list.append(t_sen)
            spec_list.append(t_spec)
            run.finish()
        return acc_list, auc_list, sen_list, spec_list
    else:
        with open_dict(cfg):
            cfg.unique_id = f"{cfg.optimizer.base_lr}_{cfg.model.sc_loss_lamda}_{cfg.model.pc_loss_lamda}_{SEED}"
        logger = logger_factory(cfg)
        model = model_factory(cfg)
        if cfg.state == 'testing':
            print("############testing##############")
            cfg.training.epochs = 1
            save_dir = Path(cfg.log_path) / cfg.unique_id
            model_weight_path = save_dir / "model.pt"
            if not os.path.exists(model_weight_path):
                raise FileNotFoundError(f"模型权重文件不存在：{model_weight_path}")
            state_dict = torch.load(model_weight_path)
            model.load_state_dict(state_dict)
            logger.info(f"成功加载模型权重：{model_weight_path}")
        optimizers = optimizers_factory(
            model=model, optimizer_configs=cfg.optimizer)
        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
        training = training_factory(cfg, model, optimizers,
                                    lr_schedulers, dataloaders, logger)

        t_acc, t_auc, t_sen, t_spec = training.train()
        return t_acc, t_auc, t_sen, t_spec






def generate_hyperparameter_grid(param_grid):
    """从配置生成超参数网格"""
    param_grid_dict = OmegaConf.to_container(param_grid, resolve=True)

    # 生成所有参数组合
    keys = param_grid_dict.keys()
    values = param_grid_dict.values()
    param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return param_combinations


def run_single_experiment(cfg: DictConfig, params=None):
    """运行单次实验"""
    if params:
        with open_dict(cfg):
            for key, value in params.items():
                if '.' in key:
                    parts = key.split('.')
                    current = cfg
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], value)
                else:
                    setattr(cfg, key, value)
    if '{wsr}' in cfg.dataset.path:
        cfg.dataset.path = cfg.dataset.path.format(wsr=cfg.dataset.wsr)
    with open_dict(cfg):
        cfg.group_name = f"{cfg.dataset.task}"

    acc_list = []
    auc_list = []
    sen_list = []
    spec_list = []
    # 随机种子
    if cfg.get('kfold', {}).get('enable', False):
        seeds = [42]
    else:
        seeds = [0,1,2,3,4]
    for it in seeds:
        SEED = it
        print("SEED =",SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        if cfg.get('kfold', {}).get('enable', False):
            t_acc, t_auc, t_sen, t_spec = model_training(cfg, SEED, params)
            acc_list=t_acc
            auc_list=t_auc
            sen_list=t_sen
            spec_list=t_spec

        else:
            run = wandb.init(
                project=cfg.project,
                group=f"{cfg.group_name}",
                name=f"{cfg.group_name}_lr{cfg.optimizer.base_lr}_wd{cfg.optimizer.weight_decay}_{SEED}",
                reinit=True,
                # mode = "offline"
            )
            t_acc, t_auc, t_sen, t_spec = model_training(cfg, SEED, params)
            acc_list.append(t_acc)
            auc_list.append(t_auc)
            sen_list.append(t_sen)
            spec_list.append(t_spec)
            run.finish()

    result = {
        "params": params if params else "default",
        "mean_acc": np.mean(acc_list),
        "mean_auc": np.mean(auc_list),
        "mean_sen": np.mean(sen_list),
        "mean_spec": np.mean(spec_list),
        "std_acc": np.std(acc_list),
        "std_auc": np.std(auc_list),
        "std_sen": np.std(sen_list),
        "std_spec": np.std(spec_list),
    }

    return result, acc_list, auc_list, sen_list, spec_list


def run_grid_search(cfg: DictConfig, param_grid_config):
    """运行网格搜索"""
    param_combinations = generate_hyperparameter_grid(param_grid_config)
    all_results = []
    for i, params in enumerate(param_combinations):
        print(f"\n=== 超参数组合 {i + 1}/{len(param_combinations)} ===")
        print(f"参数: {params}")
        result, _, _, _, _ = run_single_experiment(cfg, params)
        all_results.append(result)
        results_file = "grid_search_results.json"
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"测试准确率: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
        print(f"测试AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    if all_results:
        best_result = max(all_results, key=lambda x: x["mean_acc"])
        print("\n=== 最佳超参数组合 ===")
        print(f"参数: {best_result['params']}")
        print(f"测试准确率: {best_result['mean_acc']:.4f} ± {best_result['std_acc']:.4f}")
        print(f"测试AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
        print(f"测试敏感度: {best_result['mean_sen']:.4f} ± {best_result['std_sen']:.4f}")
        print(f"测试特异性: {best_result['mean_spec']:.4f} ± {best_result['std_spec']:.4f}")
    results_file = "grid_search_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    return all_results
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    if hasattr(cfg, 'grid_search') and getattr(cfg.grid_search, 'enable', False):
        print("开始网格搜索...")
        if not hasattr(cfg.grid_search, 'config_file') or cfg.grid_search.config_file is None:
            print("错误：未指定网格搜索配置文件！")
            return
        grid_search_config_path = os.path.join("conf", "grid_search", cfg.grid_search.config_file)
        if not os.path.exists(grid_search_config_path):
            print(f"错误：网格搜索配置文件不存在：{grid_search_config_path}")
            return
        try:

            param_grid_config = OmegaConf.load(grid_search_config_path)
            print(f"已加载网格搜索配置：{cfg.grid_search.config_file}")
            all_results = run_grid_search(cfg, param_grid_config)
        except Exception as e:
            print(f"加载或运行网格搜索时出错：{e}")
            return

    else:
        print("运行单次实验...")
        result, acc_list, auc_list, sen_list, spec_list = run_single_experiment(cfg)

        print(f"\n=== 实验结果 ===")
        print(f"测试准确率: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
        print(f"测试AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
        print(f"测试敏感度: {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
        print(f"测试特异性: {np.mean(spec_list):.4f} ± {np.std(spec_list):.4f}")
        results_file = "single_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump([result], f, indent=4)

if __name__ == '__main__':
    main()