import gzip
from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pickle

class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.state = cfg.state
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.device = next(model.parameters()).device
        self.init_meters()
        self.mseLoss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()
        self.loss_discriminator_func = torch.nn.BCELoss()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        if self.state == 'testing':
            aligned_feat0_train=[]
            aligned_feat1_train=[]
            intra_attn0_train=[]
            intra_attn1_train=[]
            cross_att_0_1_0_train=[]
            cross_att_1_0_1_train=[]
            combined_attn0_train=[]
            combined_attn1_train=[]
            feat_atlas0_batch_train=[]
            feat_atlas1_batch_train=[]
            overlap_0_1_train=[]
            overlap_1_0_train=[]
            h_stacked_0_train =[]
            h_stacked_1_train =[]
            mp_stacked_0_train =[]
            mp_stacked_1_train =[]
            all_features_0_train =[]
            all_features_1_train =[]

            labels_train=[]
            correctness_train = []
        if self.state=='training':
            self.model.train()
        for batched_graphs,batched_labels in self.train_dataloader:
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            batched_graphs = batched_graphs.to(self.device)
            batched_labels = batched_labels.float().to(self.device)
            predict,sc_loss,PC_loss,result = self.model(batched_graphs)
            cross_loss = self.loss_fn(predict, batched_labels)
            supervised_loss=cross_loss+self.config.model.sc_loss_lamda*sc_loss+self.config.model.pc_loss_lamda*PC_loss
            self.train_loss.update_with_weight(supervised_loss.item(), batched_labels.shape[0])
            if self.state == 'training':
                optimizer.zero_grad()
                supervised_loss.backward()
                optimizer.step()
            top1 = accuracy(predict, batched_labels[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, batched_labels.shape[0])
            label = batched_labels
            if self.state == 'testing':
                aligned_feat0_train.append(result[0].detach().cpu().numpy())
                aligned_feat1_train.append(result[1].detach().cpu().numpy())
                intra_attn0_train.append(result[2].detach().cpu().numpy())
                intra_attn1_train.append(result[3].detach().cpu().numpy())
                cross_att_0_1_0_train.append(result[4].detach().cpu().numpy())
                cross_att_1_0_1_train.append(result[5].detach().cpu().numpy())
                combined_attn0_train.append(result[6].detach().cpu().numpy())
                combined_attn1_train.append(result[7].detach().cpu().numpy())
                feat_atlas0_batch_train.append(result[8].detach().cpu().numpy())
                feat_atlas1_batch_train.append(result[9].detach().cpu().numpy())
                overlap_0_1_train.append(result[10].detach().cpu().numpy())
                overlap_1_0_train.append(result[11].detach().cpu().numpy())
                h_stacked_0_train.append(result[12].detach().cpu().numpy())
                h_stacked_1_train.append(result[13].detach().cpu().numpy())
                mp_stacked_0_train.append(result[14].detach().cpu().numpy())
                mp_stacked_1_train.append(result[15].detach().cpu().numpy())
                all_features_0_train.append(result[16].detach().cpu().numpy())
                all_features_1_train.append(result[17].detach().cpu().numpy())
                labels_train.append(label.detach().cpu().numpy())

                pred_class = torch.argmax(predict, dim=1)
                true_class = batched_labels[:, 1].long()
                correct = (pred_class == true_class).float().cpu().numpy()
                correctness_train.append(correct)
        if self.state == 'testing':
            np.save(self.save_path/f"aligned_feat0_train.npy", np.concatenate(aligned_feat0_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"aligned_feat1_train.npy", np.concatenate(aligned_feat1_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn0_train.npy", np.concatenate(intra_attn0_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn1_train.npy", np.concatenate(intra_attn1_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_0_1_0_train.npy", np.concatenate(cross_att_0_1_0_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_1_0_1_train.npy", np.concatenate(cross_att_1_0_1_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn0_train.npy", np.concatenate(combined_attn0_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn1_train.npy", np.concatenate(combined_attn1_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas0_batch_train.npy", np.concatenate(feat_atlas0_batch_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas1_batch_train.npy", np.concatenate(feat_atlas1_batch_train, axis=0), allow_pickle=True)
            np.save(self.save_path / f"overlap_0_1_train.npy", np.concatenate(overlap_0_1_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"overlap_1_0_train.npy", np.concatenate(overlap_1_0_train, axis=0),allow_pickle=True)

            np.save(self.save_path / f"h_stacked_0_train.npy", np.concatenate(h_stacked_0_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"h_stacked_1_train.npy", np.concatenate(h_stacked_1_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_0_train.npy", np.concatenate(mp_stacked_0_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_1_train.npy", np.concatenate(mp_stacked_1_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"all_features_0_train.npy", np.concatenate(all_features_0_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"all_features_1_train.npy", np.concatenate(all_features_1_train, axis=0),allow_pickle=True)
            np.save(self.save_path / f"labels_train.npy", np.concatenate(labels_train, axis=0),allow_pickle=True)
            # 新增保存正确性
            np.save(self.save_path / f"correctness_train.npy", np.concatenate(correctness_train, axis=0),allow_pickle=True)

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        out = []
        if self.state == 'testing':
            aligned_feat0_test=[]
            aligned_feat1_test=[]
            intra_attn0_test=[]
            intra_attn1_test=[]
            cross_att_0_1_0_test=[]
            cross_att_1_0_1_test=[]
            combined_attn0_test=[]
            combined_attn1_test=[]
            feat_atlas0_batch_test=[]
            feat_atlas1_batch_test=[]
            overlap_0_1_test=[]
            overlap_1_0_test=[]

            h_stacked_0_test = []
            h_stacked_1_test = []
            mp_stacked_0_test = []
            mp_stacked_1_test = []
            all_features_0_test = []
            all_features_1_test = []
            labels_test=[]
            correctness_test = []
        self.model.eval()

        for batched_graphs, batched_labels in dataloader:
            batched_graphs = batched_graphs.to(self.device)
            batched_labels = batched_labels.float().to(self.device)
            predict,sc_loss,PC_loss,result = self.model(batched_graphs)

            cross_loss = self.loss_fn(predict, batched_labels)
            supervised_loss=cross_loss+self.config.model.sc_loss_lamda*sc_loss+self.config.model.pc_loss_lamda*PC_loss
            loss_meter.update_with_weight(
                supervised_loss.item(), batched_labels.shape[0])
            top1 = accuracy(predict, batched_labels[:, 1])[0]
            acc_meter.update_with_weight(top1, batched_labels.shape[0])
            out += F.softmax(predict, dim=1)[:, 1].tolist()
            labels += batched_labels[:, 1].tolist()
            label = batched_labels
            if self.state == 'testing':
                aligned_feat0_test.append(result[0].detach().cpu().numpy())
                aligned_feat1_test.append(result[1].detach().cpu().numpy())
                intra_attn0_test.append(result[2].detach().cpu().numpy())
                intra_attn1_test.append(result[3].detach().cpu().numpy())
                cross_att_0_1_0_test.append(result[4].detach().cpu().numpy())
                cross_att_1_0_1_test.append(result[5].detach().cpu().numpy())
                combined_attn0_test.append(result[6].detach().cpu().numpy())
                combined_attn1_test.append(result[7].detach().cpu().numpy())
                feat_atlas0_batch_test.append(result[8].detach().cpu().numpy())
                feat_atlas1_batch_test.append(result[9].detach().cpu().numpy())
                overlap_0_1_test.append(result[10].detach().cpu().numpy())
                overlap_1_0_test.append(result[11].detach().cpu().numpy())
                h_stacked_0_test.append(result[12].detach().cpu().numpy())
                h_stacked_1_test.append(result[13].detach().cpu().numpy())
                mp_stacked_0_test.append(result[14].detach().cpu().numpy())
                mp_stacked_1_test.append(result[15].detach().cpu().numpy())
                all_features_0_test.append(result[16].detach().cpu().numpy())
                all_features_1_test.append(result[17].detach().cpu().numpy())
                labels_test.append(label.detach().cpu().numpy())
                # 新增正确性
                pred_class = torch.argmax(predict, dim=1)
                true_class = batched_labels[:, 1].long()
                correct = (pred_class == true_class).float().cpu().numpy()
                correctness_test.append(correct)
        if self.state == 'testing':
            np.save(self.save_path / f"aligned_feat0_test.npy", np.concatenate(aligned_feat0_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"aligned_feat1_test.npy", np.concatenate(aligned_feat1_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn0_test.npy", np.concatenate(intra_attn0_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn1_test.npy", np.concatenate(intra_attn1_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_0_1_0_test.npy", np.concatenate(cross_att_0_1_0_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_1_0_1_test.npy", np.concatenate(cross_att_1_0_1_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn0_test.npy", np.concatenate(combined_attn0_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn1_test.npy", np.concatenate(combined_attn1_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas0_batch_test.npy", np.concatenate(feat_atlas0_batch_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas1_batch_test.npy", np.concatenate(feat_atlas1_batch_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"overlap_0_1_test.npy", np.concatenate(overlap_0_1_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"overlap_1_0_test.npy", np.concatenate(overlap_1_0_test, axis=0), allow_pickle=True)
            np.save(self.save_path / f"labels_test.npy", np.concatenate(labels_test, axis=0),allow_pickle=True)
            np.save(self.save_path / f"correctness_test.npy", np.concatenate(correctness_test, axis=0), allow_pickle=True)

            np.save(self.save_path / f"h_stacked_0_test.npy", np.concatenate(h_stacked_0_test, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"h_stacked_1_test.npy", np.concatenate(h_stacked_1_test, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_0_test.npy", np.concatenate(mp_stacked_0_test, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_1_test.npy", np.concatenate(mp_stacked_1_test, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"all_features_0_test.npy", np.concatenate(all_features_0_test, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"all_features_1_test.npy", np.concatenate(all_features_1_test, axis=0),
                    allow_pickle=True)

        auc = roc_auc_score(labels, out)
        out, labels = np.array(out), np.array(labels)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, out, average='micro')

        report = classification_report(
            labels, out, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return [auc] + list(metric) + recall

    def val_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        out = []
        if self.state == 'testing':
            aligned_feat0_val=[]
            aligned_feat1_val=[]
            intra_attn0_val=[]
            intra_attn1_val=[]
            cross_att_0_1_0_val=[]
            cross_att_1_0_1_val=[]
            combined_attn0_val=[]
            combined_attn1_val=[]
            feat_atlas0_batch_val=[]
            feat_atlas1_batch_val=[]
            overlap_0_1_val=[]
            overlap_1_0_val=[]
            h_stacked_0_val = []
            h_stacked_1_val = []
            mp_stacked_0_val = []
            mp_stacked_1_val = []
            all_features_0_val = []
            all_features_1_val = []
            labels_val = []
            correctness_val = []  # 新增
        self.model.eval()

        for batched_graphs, batched_labels in dataloader:
            batched_graphs = batched_graphs.to(self.device)
            batched_labels = batched_labels.float().to(self.device)
            predict,sc_loss,PC_loss,result= self.model(batched_graphs)

            cross_loss = self.loss_fn(predict, batched_labels)
            supervised_loss=cross_loss+self.config.model.sc_loss_lamda*sc_loss+self.config.model.pc_loss_lamda*PC_loss #+align_loss
            loss_meter.update_with_weight(
                supervised_loss.item(), batched_labels.shape[0])
            top1 = accuracy(predict, batched_labels[:, 1])[0]
            acc_meter.update_with_weight(top1, batched_labels.shape[0])
            out += F.softmax(predict, dim=1)[:, 1].tolist()
            labels += batched_labels[:, 1].tolist()
            label=batched_labels
            if self.state == 'testing':
                aligned_feat0_val.append(result[0].detach().cpu().numpy())
                aligned_feat1_val.append(result[1].detach().cpu().numpy())
                intra_attn0_val.append(result[2].detach().cpu().numpy())
                intra_attn1_val.append(result[3].detach().cpu().numpy())
                cross_att_0_1_0_val.append(result[4].detach().cpu().numpy())
                cross_att_1_0_1_val.append(result[5].detach().cpu().numpy())
                combined_attn0_val.append(result[6].detach().cpu().numpy())
                combined_attn1_val.append(result[7].detach().cpu().numpy())
                feat_atlas0_batch_val.append(result[8].detach().cpu().numpy())
                feat_atlas1_batch_val.append(result[9].detach().cpu().numpy())
                overlap_0_1_val.append(result[10].detach().cpu().numpy())
                overlap_1_0_val.append(result[11].detach().cpu().numpy())

                h_stacked_0_val.append(result[12].detach().cpu().numpy())
                h_stacked_1_val.append(result[13].detach().cpu().numpy())
                mp_stacked_0_val.append(result[14].detach().cpu().numpy())
                mp_stacked_1_val.append(result[15].detach().cpu().numpy())
                all_features_0_val.append(result[16].detach().cpu().numpy())
                all_features_1_val.append(result[17].detach().cpu().numpy())

                labels_val.append(label.detach().cpu().numpy())
                # 新增正确性
                pred_class = torch.argmax(predict, dim=1)
                true_class = batched_labels[:, 1].long()
                correct = (pred_class == true_class).float().cpu().numpy()
                correctness_val.append(correct)
        if self.state == 'testing':
            np.save(self.save_path / f"aligned_feat0_val.npy", np.concatenate(aligned_feat0_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"aligned_feat1_val.npy", np.concatenate(aligned_feat1_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn0_val.npy", np.concatenate(intra_attn0_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"intra_attn1_val.npy", np.concatenate(intra_attn1_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_0_1_0_val.npy", np.concatenate(cross_att_0_1_0_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"cross_att_1_0_1_val.npy", np.concatenate(cross_att_1_0_1_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn0_val.npy", np.concatenate(combined_attn0_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"combined_attn1_val.npy", np.concatenate(combined_attn1_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas0_batch_val.npy", np.concatenate(feat_atlas0_batch_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"feat_atlas1_batch_val.npy", np.concatenate(feat_atlas1_batch_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"overlap_0_1_val.npy", np.concatenate(overlap_0_1_val, axis=0), allow_pickle=True)
            np.save(self.save_path / f"overlap_1_0_val.npy", np.concatenate(overlap_1_0_val, axis=0), allow_pickle=True)

            np.save(self.save_path / f"labels_val.npy", np.concatenate(labels_val, axis=0),allow_pickle=True)
            # 新增保存
            np.save(self.save_path / f"correctness_val.npy", np.concatenate(correctness_val, axis=0), allow_pickle=True)

            np.save(self.save_path / f"h_stacked_0_val.npy", np.concatenate(h_stacked_0_val, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"h_stacked_1_val.npy", np.concatenate(h_stacked_1_val, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_0_val.npy", np.concatenate(mp_stacked_0_val, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"mp_stacked_1_val.npy", np.concatenate(mp_stacked_1_val, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"all_features_0_val.npy", np.concatenate(all_features_0_val, axis=0),
                    allow_pickle=True)
            np.save(self.save_path / f"all_features_1_val.npy", np.concatenate(all_features_1_val, axis=0),
                    allow_pickle=True)
        auc = roc_auc_score(labels, out)
        out, labels = np.array(out), np.array(labels)
        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, out, average='micro')

        report = classification_report(
            labels, out, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return [auc] + list(metric) + recall



    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_acc = 0
        best_test_acc = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            if self.val_dataloader is not None:
                val_result = self.val_per_epoch(self.val_dataloader,
                                                self.val_loss, self.val_accuracy)
            else:
                val_result = None

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spe:{test_result[-2]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

        if self.state == 'training':
            self.save_result(training_process)
            test_acc = self.test_accuracy.avg
            test_AUC = test_result[0]
            test_sen = test_result[-1]
            test_spec = test_result[-2]
            wandb.run.summary["Test Accuracy"] = self.test_accuracy.avg
            wandb.run.summary["Test AUC"] = test_result[0]
            wandb.run.summary["Test Sensitivity"] = test_result[-1]
            wandb.run.summary["Test Specificity"] = test_result[-2]
        return [test_acc,test_AUC,test_sen,test_spec]
