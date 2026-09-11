import torch
from dgl.nn.pytorch import GraphConv, SGConv
import torch as th
from torch import nn
from torch.nn import init
from dgl import laplacian_lambda_max, broadcast_nodes, function as fn
import torch.nn.functional as F

from .loss import population_graph, MSE_loss, Contrastive_loss
from .readout import HeteroGraphReadout
from ..base import BaseModel
import dgl
import numpy as np
import random
from omegaconf import DictConfig

class OverlapMatrixExtractor(nn.Module):
    """
    从异质图中提取重叠矩阵C和D的模块

    C: 图谱0到图谱1的重叠矩阵，形状(N0, N1)
    D: 图谱1到图谱0的重叠矩阵，形状(N1, N0)

    注意：这里假设每个batch中的每个样本的脑区数相同
    """

    def __init__(self, normalization='row', eps=1e-8):
        """
        参数:
            normalization: 归一化方式
                - 'row': 行归一化，每行和为1。这会将绝对重叠转换为相对概率分布
                - 'none': 不归一化，保留绝对重叠数值
                **注意**: 即使原始数值在0-1之间，归一化仍然有意义：
                    * 行归一化：使每行和为1，便于解释为概率转移
                    * 不归一化：保留原始重叠的绝对规模信息
            eps: 避免除零的小量
        """
        super().__init__()
        self.normalization = normalization
        self.eps = eps

    def forward(self, g):
        """
        从异质图中提取重叠矩阵C和D

        参数:
            g: 异质图，必须包含边('atlas0', 'atlas0-atlas1', 'atlas1')和
                         ('atlas1', 'atlas1-atlas0', 'atlas0')

        返回:
            C_list: 每个样本的C矩阵列表
            D_list: 每个样本的D矩阵列表
        """
        device = g.device

        batch_size = g.batch_size
        num_regions_atlas0 = g.batch_num_nodes('atlas0')
        num_regions_atlas1 = g.batch_num_nodes('atlas1')

        if batch_size == 0:
            return [], []

        N0_per_sample = num_regions_atlas0[0].item()
        N1_per_sample = num_regions_atlas1[0].item()

        edges_0_to_1 = g.edges(etype=('atlas0', 'atlas0-atlas1', 'atlas1'))
        edges_1_to_0 = g.edges(etype=('atlas1', 'atlas1-atlas0', 'atlas0'))

        weights_0_to_1 = g.edges[('atlas0', 'atlas0-atlas1', 'atlas1')].data['w']
        weights_1_to_0 = g.edges[('atlas1', 'atlas1-atlas0', 'atlas0')].data['w']

        C_list = []
        D_list = []
        for batch_idx in range(batch_size):
            offset_atlas0 = batch_idx * N0_per_sample
            offset_atlas1 = batch_idx * N1_per_sample
            C = self._extract_matrix_for_sample(
                edges_0_to_1, weights_0_to_1,
                offset_atlas0, offset_atlas1,
                N0_per_sample, N1_per_sample,
                device
            )
            D = self._extract_matrix_for_sample(
                edges_1_to_0, weights_1_to_0,
                offset_atlas1, offset_atlas0,
                N1_per_sample, N0_per_sample,
                device
            )
            C_list.append(C)
            D_list.append(D)
        C_batch = torch.stack(C_list, dim=0)
        D_batch = torch.stack(D_list, dim=0)
        return C_batch, D_batch

    def _extract_matrix_for_sample(self, edges, weights, offset_src, offset_dst,
                                   N_src, N_dst, device):
        """为单个样本提取矩阵"""
        matrix = torch.zeros(N_src, N_dst, device=device)

        mask = ((edges[0] >= offset_src) & (edges[0] < offset_src + N_src) &
                (edges[1] >= offset_dst) & (edges[1] < offset_dst + N_dst))

        if mask.sum() > 0:
            src_idx = edges[0][mask] - offset_src
            dst_idx = edges[1][mask] - offset_dst
            edge_weights = weights[mask]

            # 填充矩阵
            matrix[src_idx, dst_idx] = edge_weights

        return matrix


class IntraAtlasAttention(nn.Module):
    """
    图谱内部注意力模块
    使用原始维度计算图谱内节点之间的注意力得分矩阵
    """

    def __init__(self, embed_dim,reduced_dim,att_smoothing,alpha, num_heads=4, dropout=0.1,residual=False,norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = reduced_dim // num_heads
        self.num_heads = num_heads
        self.att_smoothing = att_smoothing
        self.residual=residual
        self.norm=norm
        self.alpha = alpha
        if self.att_smoothing ==0:
            self.alpha = nn.Parameter(torch.tensor(self.alpha))
        assert reduced_dim % num_heads == 0, f"reduced_dim {reduced_dim} must be divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(embed_dim, reduced_dim)
        self.k_proj = nn.Linear(embed_dim, reduced_dim)
        self.v_proj = nn.Linear(embed_dim, reduced_dim)

        self.out_proj = nn.Linear(reduced_dim, reduced_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(reduced_dim)

    def forward(self, x,overlap_a_b,overlap_b_a):
        """
        输入: (batch_size, num_nodes, embed_dim)
        输出: 增强特征, 注意力矩阵 (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape
        overlap_a_b_expanded = overlap_a_b.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        overlap_b_a_expanded = overlap_b_a.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # 计算Q, K, V
        Q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch, heads, nodes, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        intra_attn = torch.softmax(scores, dim=-1)
        cross_att = torch.matmul(intra_attn, overlap_a_b_expanded)  # N0*N1
        cross_att_cycle = torch.matmul(cross_att, overlap_b_a_expanded)  # N0*N0

        if self.att_smoothing == 0:
            smoothed_attn = self.alpha * intra_attn + (1 - self.alpha) * cross_att_cycle
        elif self.att_smoothing == 1:
            smoothed_attn = self.alpha * intra_attn + (1 - self.alpha) * cross_att_cycle
        else:
            combined_attn = intra_attn + cross_att_cycle
            smoothed_attn = F.softmax(combined_attn, dim=-1)
        attended = torch.matmul(smoothed_attn, V)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_nodes, -1)
        attended = self.out_proj(attended) ################################
        if self.residual:
            attended=attended + x
        if self.norm:
            attended = self.layer_norm(attended)

        attended = self.dropout(attended)
        return attended,intra_attn,cross_att_cycle,combined_attn #,cycle_loss


class CrossAtlasCycleAttentionAlignment(nn.Module):
    """
    使用跨图谱边权重进行注意力对齐
    步骤：
    1. 分别计算两个图谱的内部注意力矩阵
    2. 使用跨图谱边权重对齐两个注意力矩阵
    3. 将对齐后的注意力矩阵应用到各自的特征上
    """

    def __init__(self, dim_atlas0, dim_atlas1, aligned_dim, num_heads=4, dropout=0.1,
                 use_cycle_consistency=True,att_smoothing=0, alpha=0.7):
        super().__init__()
        self.aligned_dim = aligned_dim
        self.use_cycle_consistency = use_cycle_consistency
        self.intra_attention_atlas0 = IntraAtlasAttention(
            embed_dim=dim_atlas0,
            reduced_dim=aligned_dim,
            att_smoothing=att_smoothing,
            alpha=alpha,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.intra_attention_atlas1 = IntraAtlasAttention(
            embed_dim=dim_atlas1,
            reduced_dim=aligned_dim,
            att_smoothing=att_smoothing,
            alpha=alpha,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feat_proj_atlas0 = nn.Linear(dim_atlas0, aligned_dim)
        self.feat_proj_atlas1 = nn.Linear(dim_atlas1, aligned_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_0 = nn.LayerNorm(aligned_dim)
        self.layer_norm_1 = nn.LayerNorm(aligned_dim)
        self.Contrastive_loss = Contrastive_loss()


    def forward(self, feat_atlas0, feat_atlas1,overlap_0_1,overlap_1_0, g):
        """
        前向传播：
        1. 计算各自内部注意力
        2. 对齐注意力矩阵
        3. 将注意力应用到特征上
        4. 投影到统一维度
        5. （可选）应用循环一致性对齐
        feat_atlas0:N1*N1
        feat_atlas1:N2*N2
        overlap_0_1:N1*N2
        overlap_1_0:N2*N1
        """

        batch_size = g.batch_size
        num_regions_atlas0 = g.batch_num_nodes('atlas0')
        num_regions_atlas1 = g.batch_num_nodes('atlas1')

        N0_per_sample = num_regions_atlas0[0].item() if batch_size > 0 else 0
        N1_per_sample = num_regions_atlas1[0].item() if batch_size > 0 else 0

        if batch_size > 0 and N0_per_sample > 0:
            feat_atlas0_batch = feat_atlas0.view(batch_size, N0_per_sample, -1)
            enhanced_feat0,intra_attn0,cross_att_0_1_0,combined_attn0 = self.intra_attention_atlas0(feat_atlas0_batch,overlap_0_1,overlap_1_0)
        else:
            batch_size = 1
            feat_atlas0_batch = feat_atlas0.unsqueeze(0)
            enhanced_feat0,intra_attn0,cross_att_0_1_0,combined_attn0 = self.intra_attention_atlas0(feat_atlas0_batch,overlap_0_1,overlap_1_0)
            N0_per_sample = feat_atlas0.shape[0]

        if batch_size > 0 and N1_per_sample > 0:
            feat_atlas1_batch = feat_atlas1.view(batch_size, N1_per_sample, -1)
            enhanced_feat1, intra_attn1, cross_att_1_0_1,combined_attn1 = self.intra_attention_atlas1(feat_atlas1_batch,overlap_1_0,overlap_0_1)
        else:
            feat_atlas1_batch = feat_atlas1.unsqueeze(0)
            enhanced_feat1, intra_attn1, cross_att_1_0_1,combined_attn1 = self.intra_attention_atlas1(feat_atlas1_batch,overlap_1_0,overlap_0_1)
            N1_per_sample = feat_atlas1.shape[0]
        sc_loss = self.Contrastive_loss(enhanced_feat0, enhanced_feat1)
        return sc_loss,enhanced_feat0,enhanced_feat1,intra_attn0,intra_attn1,cross_att_0_1_0,cross_att_1_0_1,combined_attn0,combined_attn1,feat_atlas0_batch,feat_atlas1_batch



class SGCLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=True):
        super(SGCLayer, self).__init__()
        self.gcn_layer = SGConv(in_feats,
                                out_feats,
                                k,
                                cached,
                                bias,
                                norm,
                                allow_zero_in_degree)
        self.res = nn.Linear(in_feats, out_feats)

    def forward(self, bg, feat):
        return self.gcn_layer(bg, feat) + self.res(feat)


class HeteroGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 weighted=False,
                 activation=F.relu,
                 dropout=0.0):
        super(HeteroGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._weighted = weighted

        self.send = fn.copy_u('h', 'm')
        self.recv = fn.sum(msg='m', out='h')
        self.edge_func = nn.Linear(1, out_feats)
        if self._weighted: #如果是加权图
            self.send = fn.u_mul_e('h', 'w_feat', 'm')

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation
        self.bn = nn.BatchNorm1d(out_feats)

        self.dropout = nn.Dropout(dropout)


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, input_feat, etype):
        graph = graph.local_var()
        graph.edges[etype].data['w_feat'] = self.edge_func(graph.edges[etype].data['w'].reshape(-1, 1))
        src_type = etype[:etype.index('-')]
        dst_type = etype[etype.index('-') + 1:]

        # mult W first to reduce the feature size for aggregation.
        feat = th.matmul(input_feat, self.weight)
        graph.nodes[src_type].data['h'] = feat
        graph.update_all(self.send, self.recv, etype=etype)
        rst = graph.nodes[dst_type].data['h']

        if self.bias is not None:
            rst = rst + self.bias
        if self.activation is not None:
            rst = self.activation(rst)

        new_feats = self.dropout(rst)
        return new_feats

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes, activation=F.relu, weighted=False, bias=False, mode='hetero'):
        super(HeteroRGCNLayer, self).__init__()
        self.mode = mode
        if self.mode == 'hetero':
            self.gcn_layers = nn.ModuleDict({
                    etype: HeteroGraphConv(in_feats=in_feats, out_feats=out_feats, activation=None,
                                           weighted=weighted, bias=bias,
                                           ) for etype in etypes
                })
        else:
            assert self.mode == 'homo'
            conv = HeteroGraphConv(in_feats=in_feats, out_feats=out_feats, activation=None,
                                           weighted=weighted, bias=bias,
                                           )

            self.gcn_layers = nn.ModuleDict({
                'l-r': conv,
                'l-l': conv
            })
        self.activation = activation

        self.etypes = etypes

    def forward(self, G, feat_dict, R_aggregate_mode='mean'):
        G = G.local_var()
        for etype in self.etypes:
            if etype == 'atlas0-atlas1':
                for (srctype, dsttype) in [('atlas0', 'atlas1')]:

                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['cross'] = dst_etype_feat
            elif etype == 'atlas1-atlas0':
                for (srctype, dsttype) in [('atlas1', 'atlas0')]:

                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['cross'] = dst_etype_feat
            elif etype == 'atlas0-atlas0':
                for (srctype, dsttype) in [('atlas0', 'atlas0')]:
                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['intra'] = dst_etype_feat
            elif etype == 'atlas1-atlas1':
                for (srctype, dsttype) in [('atlas1', 'atlas1')]:
                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['intra'] = dst_etype_feat


        new_feature_dict = {}
        for ntype in G.ntypes:
            if G.num_nodes(ntype) != 0:
                ntype_feats = [G.nodes[ntype].data['cross'], G.nodes[ntype].data['intra']]
                ntype_feat = th.mean(th.stack(ntype_feats), 0)
                if self.activation is not None:
                    ntype_feat = self.activation(ntype_feat)
                new_feature_dict[ntype] = ntype_feat

        return new_feature_dict


class AP_CHGRL(BaseModel):
    """
    使用跨图谱边权重进行连接对齐的模型
    注意力对齐在图卷积之前进行
    """

    def __init__(self, config: DictConfig):
        super(AP_CHGRL, self).__init__()

        self.mode = "hetero"
        self.etypes = config.dataset.etypes
        self.meta_paths = config.dataset.meta_paths
        self.in_dims_dict = config.dataset.node_dims_dict
        self.ntypes = list(self.in_dims_dict.keys())
        self.hidden_dims = config.model.hidden_dim

        self.aligned_dim = self.hidden_dims[0]

        self.overlap_extractor = OverlapMatrixExtractor(normalization='row')

        self.raw_feat_dims = self.create_uniform_dim_dict(self.ntypes, self.hidden_dims[-1])
        self.meta_feat_dims = self.create_uniform_dim_dict(self.ntypes, self.hidden_dims[-1])
        self.output_dims = self.create_uniform_dim_dict(self.ntypes, 1)

        self.feature_alignment = CrossAtlasCycleAttentionAlignment(
            dim_atlas0=self.in_dims_dict['atlas0'],
            dim_atlas1=self.in_dims_dict['atlas1'],
            aligned_dim=self.aligned_dim,
            num_heads=config.model.get('attention_heads', 4),
            dropout=config.model.get('attention_dropout', 0.1),
            use_cycle_consistency = True,
            att_smoothing = 2,
            alpha = config.model.get('alpha', 0.5)
        )

        layers = [
            HeteroRGCNLayer(
                in_feats=self.aligned_dim,
                out_feats=self.hidden_dims[0],
                etypes=self.etypes,
                activation=nn.PReLU(),
                weighted=config.model.weighted,
                bias=config.model.bias,
                mode=self.mode
            )
        ]

        if len(self.hidden_dims) >= 2:
            for i in range(1, len(self.hidden_dims)):
                layers.append(
                    HeteroRGCNLayer(
                        in_feats=self.hidden_dims[i - 1],
                        out_feats=self.hidden_dims[i],
                        etypes=self.etypes,
                        activation=nn.PReLU(),
                        weighted=config.model.weighted,
                        bias=config.model.bias,
                        mode=self.mode
                    )
                )

        self.encoder = nn.ModuleList(layers)

        self.classify_layer = nn.Sequential(
            nn.Linear(sum(self.in_dims_dict.values()), 2)
        )

        self.classify_layer1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 2)
        )
        self.ntypes_meta_path_emb = nn.ModuleDict({})

        for step, ntype in enumerate(self.ntypes):
            self.ntypes_meta_path_emb[ntype] = nn.ModuleDict({
                meta_path_name: SGCLayer(
                    in_feats=self.hidden_dims[0],
                    out_feats=self.hidden_dims[-1],
                    k=1, cached=False, bias=False,
                    norm=None, allow_zero_in_degree=True
                )
                for meta_path_name, _ in self.meta_paths[ntype]
            })


        # 6.  Readout层
        self.readout = HeteroGraphReadout(
            node_types=self.ntypes,
            raw_feat_dims=self.raw_feat_dims,
            meta_feat_dims=self.meta_feat_dims,
            output_dims=self.output_dims,
            batch_size=None
        )

        self.intra_attention_matrices = {}
        self.alignment_loss = None

        self.population_graph = population_graph()
        self.MSE_loss = MSE_loss()

    def extra_sample_features(self, g, key):
        """
        从批次异质图中提取指定 key 的特征，并重塑为 (batch, num_nodes, dim) 格式。

        参数:
            g: DGL 批次图
            key: 特征键名，如 'aligned_feat' 或 'mp_feat' 等

        返回:
            dict: 键为节点类型 (ntype)，值为形状 (batch_size, num_nodes, feature_dim) 的张量
        """
        batch_size = g.batch_size
        features_dict = {}

        for ntype in self.ntypes:
            val = g.ndata[key][ntype]
            if isinstance(val, tuple):
                feat = val[1]  # (total_nodes, dim)
            else:
                feat = val  # (total_nodes, dim)

            nodes_per_sample = g.batch_num_nodes(ntype)  # 形状 (batch_size,)

            if torch.all(nodes_per_sample == nodes_per_sample[0]):
                num_nodes = nodes_per_sample[0].item()
                feat_reshaped = feat.view(batch_size, num_nodes, -1)
            else:
                feat_split = torch.split(feat, nodes_per_sample.tolist(), dim=0)
                feat_reshaped = feat_split  # 此时为 tuple of tensors

            features_dict[ntype] = feat_reshaped

        return features_dict


    def create_uniform_dim_dict(self, node_types, dim_value):
        return {ntype: dim_value for ntype in node_types}

    def forward(self, g, k=3, method='mean'):
        """
        前向传播

        流程:
        1. 提取原始特征
        2. 用原始特征计算各自注意力矩阵，然后对齐特征维度
        3. 将对齐后的特征输入图卷积
        4. 使用原始注意力矩阵计算连接对齐损失
        5. 进行对比学习和分类
        """
        batch_size = g.batch_size
        num_regions_atlas0 = g.batch_num_nodes('atlas0')
        num_regions_atlas1 = g.batch_num_nodes('atlas1')
        N0_per_sample = num_regions_atlas0[0].item() if batch_size > 0 else 0
        N1_per_sample = num_regions_atlas1[0].item() if batch_size > 0 else 0
        original_feat_dict = {}
        for ntype in self.ntypes:
            if g.num_nodes(ntype) > 0:
                original_feat_dict[ntype] = g.ndata['feat'][ntype]
        feat_atlas0 = original_feat_dict['atlas0']
        feat_atlas1 = original_feat_dict['atlas1']

        wsr_dict = {}
        for ntype in self.ntypes:
            if g.num_nodes(ntype) > 0:
                wsr_dict[ntype] = g.ndata['wsr'][ntype]
        wsr_atlas0 = wsr_dict['atlas0']
        wsr_atlas1 = wsr_dict['atlas1']
        wsr_atlas0_batch = wsr_atlas0.view(batch_size, N0_per_sample, -1)
        wsr_atlas1_batch = wsr_atlas1.view(batch_size, N1_per_sample, -1)

        overlap_0_1,overlap_1_0=self.overlap_extractor(g)
        (sc_loss,aligned_feat0, aligned_feat1,intra_attn0,intra_attn1,cross_att_0_1_0,cross_att_1_0_1,
         combined_attn0,combined_attn1,feat_atlas0_batch,feat_atlas1_batch) \
            = self.feature_alignment(
            feat_atlas0, feat_atlas1,overlap_0_1,overlap_1_0, g)


        aligned_feat_dict = {
            'atlas0':  aligned_feat0.reshape(-1, aligned_feat0.shape[-1]),
            'atlas1': aligned_feat1.reshape(-1, aligned_feat1.shape[-1])
        }


        g.ndata['original_feat'] = original_feat_dict
        g.ndata['aligned_feat'] = aligned_feat_dict


        feat_dict = aligned_feat_dict
        for conv in self.encoder:
            feat_dict = conv(g, feat_dict)

        g.ndata['h'] = feat_dict

        for step, ntype in enumerate(self.ntypes):

            bg_ntype_meta_graphs = {
                meta_path_name: dgl.metapath_reachable_graph(g, meta_path).remove_self_loop().add_self_loop().to(g.device)
                for meta_path_name, meta_path in self.meta_paths[ntype]}
            h_pos = [self.ntypes_meta_path_emb[ntype][meta_path_name](bg_ntype_meta_graphs[meta_path_name],g.ndata['aligned_feat'][ntype]) for meta_path_name, _ in self.meta_paths[ntype]]
            g.nodes[ntype].data['mp_feat'] = h_pos[0]

        feat = self.readout(g, g.ndata['h'], g.ndata['mp_feat'])

        h_stacked = self.extra_sample_features(g,'h')
        mp_stacked = self.extra_sample_features(g, 'mp_feat')

        h_stacked_0 = h_stacked['atlas0']
        h_stacked_1 = h_stacked['atlas1']
        mp_stacked_0 = mp_stacked['atlas0']
        mp_stacked_1 = mp_stacked['atlas1']

        all_features_0 = torch.cat([h_stacked_0, mp_stacked_0], dim=2)
        all_features_1 = torch.cat([h_stacked_1, mp_stacked_1], dim=2)
        prediction = self.classify_layer(feat)


        atlas0_readout = feat[:,:self.in_dims_dict['atlas0']]
        atlas1_readout = feat[:, :self.in_dims_dict['atlas1']]
        atlas0_pg = self.population_graph(atlas0_readout)
        atlas1_pg = self.population_graph(atlas1_readout)
        PC_loss = self.MSE_loss(atlas0_pg, atlas1_pg)

        return prediction,sc_loss,PC_loss,[aligned_feat0, aligned_feat1,intra_attn0,intra_attn1,cross_att_0_1_0,cross_att_1_0_1,
         combined_attn0,combined_attn1,wsr_atlas0_batch,wsr_atlas1_batch,overlap_0_1,overlap_1_0,
                                           h_stacked_0,h_stacked_1,mp_stacked_0,mp_stacked_1,all_features_0,all_features_1]# alignment_loss

