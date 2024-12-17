import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    '''
    构建数据集
    '''
    def __init__(self, feature, label=None):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        feature_idx = self.feature[idx]
        if self.label is not None:
            label_idx = self.label[idx]
            return feature_idx, label_idx
        else:
            return feature_idx


def normalize_adjacency_matrix(adj):
    '''
    对称归一化
    adj: 邻接矩阵, shape: (N, N)
    '''
    # 计算度矩阵
    D = torch.sum(adj, dim=1)

    # 计算 D^(-1/2)
    D = torch.pow(D, -0.5)
    D = torch.diag(D)

    # 计算 D^(-1/2) * A * D^(-1/2)
    adj_normalized = torch.mm(torch.mm(D, adj), D)

    return adj_normalized
