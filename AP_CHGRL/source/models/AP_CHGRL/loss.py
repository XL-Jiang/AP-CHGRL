import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class population_graph(nn.Module):
    def __init__(self,):
        super(population_graph,self).__init__()

    def PearsonSimilarity(self,x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        xm = x - mean_x
        cov = torch.mm(xm, xm.t())/(x.shape[1] - 1)
        std_x = torch.std(x, dim=1, keepdim=True)
        pearson_matrix = cov / (torch.mm(std_x, std_x.t()) + 1e-8)
        return pearson_matrix

    def CosineSimilarity(self,x):
        x_norm = F.normalize(x, p=2, dim=1)
        similarity_matrix = torch.mm(x_norm, x_norm.t())
        return similarity_matrix

    def forward(self,H):
        #H1:[b,h]
        G = self.CosineSimilarity(H)

        return G

class MSE_loss(nn.Module):
    def __init__(self,):
        super(MSE_loss,self).__init__()
        self.mse_loss = nn.MSELoss()

    def calculate_mse(self,matrix1, matrix2):
        assert matrix1.shape == matrix2.shape, "两个矩阵的形状必须相同"
        diff = matrix1 - matrix2
        squared_diff = diff ** 2
        mse = squared_diff.mean()
        return mse
    def forward(self,G1,G2):
        batch_size = G1.shape[0]
        loss = self.calculate_mse(G1,G2)
        return loss

class Contrastive_loss(nn.Module):
    def __init__(self, temperature=0.75):
        super(Contrastive_loss, self).__init__()
        self.temperature = temperature

    def forward(self, H1, H2):
        # 输入形状 (b, n, c)
        batch_size, nodes_num, _ = H1.shape
        H1 = H1.permute(1, 0, 2)  # (n, b, c)
        H2 = H2.permute(1, 0, 2)  # (n, b, c)

        loss = 0.0
        for i in range(nodes_num):
            H1_i = H1[i]  # (b, c)
            H2_i = H2[i]  # (b, c)

            # 计算余弦相似度矩阵
            sim_matrix = F.cosine_similarity(H1_i.unsqueeze(1), H2_i.unsqueeze(0), dim=2) / self.temperature
            sim_matrix_exp = torch.exp(sim_matrix)

            # 提取正样本对的相似度（对角线）
            positive_sim = torch.diag(sim_matrix_exp).unsqueeze(1)

            # 计算负样本对的相似度总和（排除对角线）
            mask = 1 - torch.eye(batch_size, dtype=sim_matrix_exp.dtype).to(sim_matrix_exp.device)
            negative_sim_sum = torch.sum(sim_matrix_exp * mask, dim=1).unsqueeze(1)

            # 计算对比损失
            loss += -torch.log(positive_sim / (negative_sim_sum + 1e-8)).mean()

        # 平均所有节点的损失
        loss /= nodes_num
        return loss