import torch
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
import numpy as np
import torch.nn.functional as F
import dgl
import torch
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
def init_kfold_heterograph_dataloaders_no_val(
        cfg: DictConfig,
        node_features0: torch.Tensor,
        node_features1: torch.Tensor,
        wsr0:torch.Tensor,
        wsr1:torch.Tensor,
        cross_atlas_adj: torch.Tensor,
        labels: torch.Tensor,
        stratified: np.array,
        n_splits: int = 5
) -> list:
    """
    生成 n_splits 折交叉验证的 DataLoader，每折仅包含训练集和测试集（无验证集）。
    返回列表，每个元素为 (train_loader, test_loader)
    """
    stratified = labels.numpy()
    hetero_graphs = []
    num_nodes0 = node_features0.shape[1]
    num_nodes1 = node_features1.shape[1]
    for i in range(node_features1.shape[0]):
        hetero_graph = build_four_type_hetero_graph(
            cfg,
            node_feature0=node_features0[i].float(),
            node_feature1=node_features1[i].float(),
            wsr0=wsr0[i].float(),
            wsr1=wsr1[i].float(),
            full_adj_matrix=cross_atlas_adj[i],
            num_nodes0=num_nodes0,
            num_nodes1=num_nodes1,
        )
        hetero_graphs.append(hetero_graph)

    labels_onehot = F.one_hot(labels.to(torch.int64))
    n = len(hetero_graphs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_loaders = []

    for train_idx, test_idx in skf.split(np.arange(n), stratified):
        train_graphs = [hetero_graphs[i] for i in train_idx]
        test_graphs = [hetero_graphs[i] for i in test_idx]
        train_labels = labels_onehot[train_idx]
        test_labels = labels_onehot[test_idx]

        def collate_fn(batch):
            graphs, lbls = zip(*batch)
            return dgl.batch(graphs), torch.stack(lbls)

        train_dataset = list(zip(train_graphs, train_labels))
        test_dataset = list(zip(test_graphs, test_labels))

        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            drop_last=cfg.dataset.drop_last,
            collate_fn=collate_fn
        )
        test_loader = GraphDataLoader(
            test_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        all_loaders.append((train_loader, test_loader))

    first_train_size = len(all_loaders[0][0].dataset)
    with open_dict(cfg):
        cfg.steps_per_epoch = (first_train_size - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    return all_loaders

def build_four_type_hetero_graph(cfg: DictConfig,node_feature0, node_feature1,wsr0,wsr1, full_adj_matrix,
                                 num_nodes0, num_nodes1):
    """
    构建包含四种边类型的异构图
    atlas0: 100个节点
    atlas1: 200个节点
    完整邻接矩阵: 300x300 (100+200=300)
    """
    adj_00 = full_adj_matrix[:num_nodes0, :num_nodes0]
    adj_01 = full_adj_matrix[:num_nodes0, num_nodes0:num_nodes0 + num_nodes1]
    adj_10 = full_adj_matrix[num_nodes0:num_nodes0 + num_nodes1, :num_nodes0]
    adj_11 = full_adj_matrix[num_nodes0:num_nodes0 + num_nodes1, num_nodes0:num_nodes0 + num_nodes1]
    graph_data = {}
    edge_index_00 = torch.nonzero(adj_00, as_tuple=False).t()
    if edge_index_00.shape[1] > 0:
        graph_data[('atlas0', 'atlas0-atlas0', 'atlas0')] = (edge_index_00[0], edge_index_00[1])

    edge_index_01 = torch.nonzero(adj_01, as_tuple=False).t()
    if edge_index_01.shape[1] > 0:
        src_nodes = edge_index_01[0]
        dst_nodes = edge_index_01[1]
        graph_data[('atlas0', 'atlas0-atlas1', 'atlas1')] = (src_nodes, dst_nodes)

    edge_index_10 = torch.nonzero(adj_10, as_tuple=False).t()
    if edge_index_10.shape[1] > 0:
        src_nodes = edge_index_10[0]
        dst_nodes = edge_index_10[1]
        graph_data[('atlas1', 'atlas1-atlas0', 'atlas0')] = (src_nodes, dst_nodes)

    edge_index_11 = torch.nonzero(adj_11, as_tuple=False).t()
    if edge_index_11.shape[1] > 0:
        src_nodes = edge_index_11[0]
        dst_nodes = edge_index_11[1]
        graph_data[('atlas1', 'atlas1-atlas1', 'atlas1')] = (src_nodes, dst_nodes)

    num_nodes_dict = {
        'atlas0': num_nodes0,
        'atlas1': num_nodes1
    }

    hetero_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    hetero_graph.nodes['atlas0'].data['feat'] = node_feature0
    hetero_graph.nodes['atlas1'].data['feat'] = node_feature1

    hetero_graph.nodes['atlas0'].data['wsr'] = wsr0
    hetero_graph.nodes['atlas1'].data['wsr'] = wsr1
    hetero_graph=set_edge_weights(hetero_graph, adj_00, adj_01, adj_10, adj_11)

    with (open_dict(cfg)):

        cfg.dataset.etypes = ['atlas0-atlas0','atlas1-atlas1','atlas0-atlas1','atlas1-atlas0']
        cfg.dataset.meta_paths ={
            'atlas0': [('atlas0-atlas1-atlas1-atlas0',
                        [ 'atlas0-atlas1', 'atlas1-atlas1', 'atlas1-atlas0'])],
            'atlas1': [('atlas1 -atlas0-atlas0-atlas1',
                        [ 'atlas1-atlas0', 'atlas0-atlas0', 'atlas0-atlas1'])],
        }

    return hetero_graph


def set_edge_weights(hetero_graph, adj_00, adj_01, adj_10, adj_11):
    """为各种边类型设置权重"""

    if ('atlas0-atlas0') in hetero_graph.etypes:
        edge_indices = hetero_graph.edges(etype=('atlas0-atlas0'))
        weights = adj_00[edge_indices[0], edge_indices[1]]
        hetero_graph.edges[('atlas0', 'atlas0-atlas0', 'atlas0')].data['w'] = weights

    if ('atlas0-atlas1') in hetero_graph.etypes:
        edge_indices = hetero_graph.edges(etype=('atlas0-atlas1'))
        weights = adj_01[edge_indices[0], edge_indices[1]]
        hetero_graph.edges[('atlas0', 'atlas0-atlas1', 'atlas1')].data['w'] = weights

    if ( 'atlas1-atlas0') in hetero_graph.etypes:
        edge_indices = hetero_graph.edges(etype=('atlas1-atlas0'))
        weights = adj_10[edge_indices[0], edge_indices[1]]
        hetero_graph.edges[('atlas1',  'atlas1-atlas0', 'atlas0')].data['w'] = weights

    if ( 'atlas1-atlas1') in hetero_graph.etypes:
        edge_indices = hetero_graph.edges(etype=('atlas1-atlas1'))
        weights = adj_11[edge_indices[0], edge_indices[1]]
        hetero_graph.edges[('atlas1', 'atlas1-atlas1', 'atlas1')].data['w'] = weights
    return hetero_graph



