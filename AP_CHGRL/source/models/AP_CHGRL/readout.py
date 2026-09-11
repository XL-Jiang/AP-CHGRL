import torch
import torch.nn as nn
import dgl
#node_type1:ftr1:[n1,d1],ftr2:[n1,d2]
#node_type2:ftr1:[n2,d1],ftr2:[n2,d2]
#(1) stack:  dim=1,
# node_type1:ftr:[ftr1,ftr2]:[n1,d1+d2];node_type2:ftr:[ftr1,ftr2]:[n2,d1+d2]
#(2) linear: outdim=1
# node_type1:linear(ftr):[ftr1,ftr2]:[n1,1];node_type2:linear(ftr):[n2,1]
#(3) flatten: [n,1] ->[n]
#node_type1:[n1];node_type2:[n2]
#(4) cat
#graph_feature:[node_type1,node_type2]:[batch,n1+n2]
class HeteroGraphReadout(nn.Module):
    """
    异质图读出模块
    处理批次异质图，整合节点原始特征和元路径特征
    """

    def __init__(self, node_types, raw_feat_dims, meta_feat_dims, output_dims=1, batch_size=None):
        """
        初始化模块

        Args:
            node_types (list): 节点类型列表，如 ['user', 'item', 'tag']
            raw_feat_dims (dict): 各节点类型的原始特征维度，如 {'user': 64, 'item': 128, 'tag': 32}
            meta_feat_dims (dict): 各节点类型的元路径特征维度，如 {'user': 32, 'item': 64, 'tag': 16}
            output_dims (dict): 各节点类型线性变换后的输出维度
            batch_size (int, optional): 批次大小，用于展平操作
        """
        super(HeteroGraphReadout, self).__init__()

        self.node_types = node_types
        self.raw_feat_dims = raw_feat_dims
        self.meta_feat_dims = meta_feat_dims
        self.output_dims = output_dims
        self.batch_size = batch_size

        # 为每种节点类型创建线性变换层
        self.linear_layers = nn.ModuleDict()
        for ntype in node_types:
            input_dim = raw_feat_dims[ntype] + meta_feat_dims[ntype]
            output_dim = output_dims[ntype]
            self.linear_layers[ntype] = nn.Linear(input_dim, output_dim)

    def forward(self, g, raw_features_dict, meta_features_dict):
        """
        前向传播

        Args:
            g (DGLHeteroGraph): 批次异质图
            raw_features_dict (dict): 节点原始特征字典
            meta_features_dict (dict): 节点元路径特征字典

        Returns:
            dict: 处理后的节点特征字典
            torch.Tensor: 展平后的特征张量
        """

        processed_features = {}

        for ntype in self.node_types:
            if ntype not in raw_features_dict or ntype not in meta_features_dict:
                continue

            raw_feat = raw_features_dict[ntype]  # shape: [num_nodes, raw_dim]
            meta_feat = meta_features_dict[ntype]  # shape: [num_nodes, meta_dim]

            # 堆叠特征
            stacked_feat = torch.cat([raw_feat, meta_feat], dim=1)  # shape: [num_nodes, raw_dim + meta_dim]

            # 线性变换
            linear_layer = self.linear_layers[ntype]
            transformed_feat = linear_layer(stacked_feat)  # shape: [num_nodes, output_dim]

            processed_features[ntype] = transformed_feat

        # 展平特征
        flattened_features = self._flatten_features(g, processed_features)

        return flattened_features

    def _flatten_features(self, g, processed_features):
        """
        展平特征

        Args:
            g (DGLHeteroGraph): 批次异质图
            processed_features (dict): 处理后的节点特征字典

        Returns:
            torch.Tensor: 展平后的特征张量
        """
        flattened_list = []

        if hasattr(g, 'batch_size') and g.batch_size > 1:
            # 处理批次图
            for i in range(g.batch_size):
                graph_features = []
                for ntype in self.node_types:
                    if ntype in processed_features:
                        node_mask = g.batch_num_nodes(ntype)[i] if i == 0 else \
                            sum(g.batch_num_nodes(ntype)[:i])
                        num_nodes = g.batch_num_nodes(ntype)[i]

                        if num_nodes > 0:
                            node_features = processed_features[ntype][node_mask:node_mask + num_nodes]
                            graph_feature=node_features.flatten()
                            graph_features.append(graph_feature)

                if graph_features:
                    graph_combined = torch.cat(graph_features, dim=0)
                    flattened_list.append(graph_combined)


            if flattened_list:
                flattened = torch.stack(flattened_list, dim=0)
            else:
                flattened = torch.tensor([])
        else:
            graph_features = []
            for ntype in self.node_types:
                if ntype in processed_features:
                    graph_feature = torch.mean(processed_features[ntype], dim=0)
                    graph_features.append(graph_feature)

            if graph_features:
                flattened = torch.cat(graph_features, dim=0).unsqueeze(0)
            else:
                flattened = torch.tensor([])

        return flattened

    def update_batch_size(self, batch_size):
        """更新批次大小"""
        self.batch_size = batch_size