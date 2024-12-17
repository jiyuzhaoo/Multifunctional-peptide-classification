import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# 将两个矩阵 a 和 b 沿对角线方向进行组合，并用 0 填充空白空间
def adjConcat(a, b):
    """
    Combine the two matrices a,b diagonally along the diagonal direction and fill the empty space with zeros
    """
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))
    right = np.row_stack((np.zeros((lena, lenb)), b))
    result = np.hstack((left, right))
    return result


class GATLayer(nn.Module):
    """
    GAT 图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features          # 输入特征维度
        self.out_features = out_features        # 输出特征维度
        self.dropout = dropout                  # dropout概率
        self.alpha = alpha                      # 反向传播时，权重的衰减系数
        self.concat = concat                    # 是否进行拼接

        # 权重初始化
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h : 节点特征矩阵. shape:[N, in_feature]
        adj : 邻接矩阵. shape:[N, N]
        '''
        Wh = torch.mm(h, self.W)    # Wh.shape:[N, out_feature]
        e = self._prepare_attentional_mechanism_input(Wh)       # e.shape:[N, N]
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # 注意力系数（未归一化）
        attention = F.softmax(attention, dim=1)     # 对权重进行归一化，每个元素代表着边的注意力权重
        h_prime = torch.matmul(attention, Wh)       # h_prime.shape:[N, out_feature]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # 计算每个节点的原始注意力系数
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, dropout, alpha, n_heads):
        """
        参数：
        - n_feat: 输入节点特征的数量
        - n_hid: 隐藏层节点特征的数量
        - dropout: Dropout比例, 用于防止过拟合
        - alpha: LeakyReLU的斜率
        - n_heads: 注意力头的数量，表示并行的注意力机制的数量
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions1 = [GATLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        self.attentions2 = [GATLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第一层注意力网络
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)

        # 第二层注意力网络
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)

        return x


class GATModel(nn.Module):
    def __init__(self, config):
        """
        config: 配置参数，包含重要参数
        """
        super(GATModel, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.gat = GAT(n_feat=config.n_feature, n_hid=config.n_hidden, dropout=config.dropout, alpha=config.alpha, n_heads=config.n_heads)

        # 为每个标签创建一个全连接层序列，用于fine-tuning (节点编码层)
        for i in range(self.num_labels):    
            setattr(self, "FC_%d" %i, nn.Sequential(
                                      nn.Linear(in_features=config.n_feature, out_features=config.n_feature),
                                      nn.Dropout()))

        # 为每个标签创建一个分类器，输入为 n_feature 维，输出为1 (节点分类层)
        for i in range(self.num_labels):  
            setattr(self, "CLSFC_%d" %i, nn.Sequential(
                                      nn.Linear(in_features=config.n_feature, out_features=1),
                                      nn.Dropout(),
                                      ))

    def forward(self, features=None, labels=None, adj_matrix=None):
        '''
        features: [batch_size, n_feature]
        labels: [batch_size, num_labels]
        adj_matrix: [num_labels, num_labels]
        '''
        outs = []
        for i in range(self.num_labels):
            FClayer = getattr(self, "FC_%d" %i)
            y = FClayer(features)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        
        outs = torch.stack(outs, dim=0).transpose(0, 1)  # [batch_size, num_labels, n_feature]
        outs = outs.reshape(-1, self.config.n_feature)   # 重塑以使用于GAT层

        for i in range(features.shape[0]):
            if i == 0:
                end_adj_matrix = adj_matrix.cpu().numpy()
            else:
                end_adj_matrix = adjConcat(end_adj_matrix, adj_matrix.cpu().numpy())    # [batch_size x num_label, batch_size x num_label]

        end_adj_matrix = torch.tensor(end_adj_matrix).to(outs.device)

        # 通过GAT层处理outs和邻接矩阵，获取更新后的嵌入
        gat_embedding = self.gat(outs, end_adj_matrix)
        gat_embedding = gat_embedding.reshape(-1, self.num_labels, self.config.n_feature)
    
        prediction_scores = list()
        for i in range(self.num_labels):
            CLSFClayer = getattr(self, "CLSFC_%d" %i)
            y = CLSFClayer(gat_embedding[:,i,:])
            prediction_scores.append(y)

        prediction_res = torch.stack(prediction_scores, dim=1).reshape(-1,self.num_labels)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(prediction_res.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            return loss, prediction_res
        else:
            return prediction_res
