import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import logging
import numpy as np
from GAT import GATModel
from model.FGM import FGM
from torch.optim import AdamW
from utils.threshold import threshold
from utils.evaluation import evaluate
from torch.utils.data import DataLoader
from utils.log_helper import logger_init
from utils.data_helpers import MultiLabelDataset


class TrainConfig:
    def __init__(self, model_num):
        # 文件路径
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'dataset', 'MFBP')

        if model_num in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            self.train_feature_path = os.path.join(self.dataset_dir, 'train', 'esm.npy')
            self.train_label_path = os.path.join(self.dataset_dir, 'train', 'label.npy')
            self.test_feature_path = os.path.join(self.dataset_dir, 'test', 'esm.npy')
            self.test_label_path = os.path.join(self.dataset_dir, 'test', 'label.npy')
            self.data_name = 'ESM'
        elif model_num in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
            self.train_feature_path = os.path.join(self.dataset_dir, 'train', 'pt.npy')
            self.train_label_path = os.path.join(self.dataset_dir, 'train', 'label.npy')
            self.test_feature_path = os.path.join(self.dataset_dir, 'test', 'pt.npy')
            self.test_label_path = os.path.join(self.dataset_dir, 'test', 'label.npy')
            self.data_name = 'PortT5'
        else:
            self.train_feature_path = os.path.join(self.dataset_dir, 'train', 'robert.npy')
            self.train_label_path = os.path.join(self.dataset_dir, 'train', 'label.npy')
            self.test_feature_path = os.path.join(self.dataset_dir, 'test', 'robert.npy')
            self.test_label_path = os.path.join(self.dataset_dir, 'test', 'label.npy')
            self.data_name = 'RoBERTa'
        

        # 相关文件保存路径
        self.model_save_dir = os.path.join(self.project_dir, 'cache', 'models')
        self.logs_save_dir = os.path.join(self.project_dir, 'cache', 'logs')
        self.res_save_dir = os.path.join(self.project_dir, 'cache', 'result')
        
        # 参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_num in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9): 
            self.n_feature = 1280       # 输入特征维度
            self.n_hidden = 128         # 隐藏层维度
            self.n_heads = 10            # 多头注意力机制的头数
        elif model_num in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
            self.n_feature = 1024       # 输入特征维度
            self.n_hidden = 128         # 隐藏层维度
            self.n_heads = 8            # 多头注意力机制的头数
        else:
            self.n_feature = 768       # 输入特征维度
            self.n_hidden = 128         # 隐藏层维度
            self.n_heads = 6            # 多头注意力机制的头数

        self.num_labels = 5         # 标签数量
        self.dropout = 0.2          # dropout
        self.alpha = 0.2            # LeakyReLU的斜率
        self.batch_size = 32        # 批大小
        self.learning_rate = 5e-5
        self.epochs = 50
        self.model_num = model_num  

        # 文件保存路径
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}_{self.model_num}.bin')
        logger_init(log_file_name="logs", log_level=logging.INFO, log_dir=self.logs_save_dir)
        self.res_save_path = os.path.join(self.res_save_dir, f'result_{self.data_name}_{self.model_num}.txt')
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)
        
        if not os.path.exists(self.res_save_dir):
            os.makedirs(self.res_save_dir)

        logging.info("\n")
        logging.info(f"\t This is {model_num} train ")
        logging.info("\t Print the current configuration to a log file ")
        for key, value in self.__dict__.items():
            logging.info(f"\t {key} = {value}")


def train(config):
    model = GATModel(config)
    logging.info(f"{model}")

    model = model.to(config.device)
    model.train()
    fgm = FGM(model)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'gat' in n],
            "lr": config.learning_rate*20,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    dataset_train = MultiLabelDataset(np.load(config.train_feature_path), np.load(config.train_label_path))
    dataset_test = MultiLabelDataset(np.load(config.test_feature_path), np.load(config.test_label_path))
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=20)

    # 构建全连接图，边权重为1 (邻接矩阵)
    graph_info = np.ones((config.num_labels, config.num_labels))
    diagnoal = np.diag([1 for i in range(config.num_labels)])      # 创建对角矩阵
    graph_info = graph_info - diagnoal
    adj_matrix = torch.from_numpy(graph_info).float().to(config.device)

    for epoch in range(config.epochs):
        model.train()
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_loader):
            sample = sample.to(config.device, dtype=torch.float32)
            label = label.to(config.device)
            
            # 前向传播
            loss, predict = model(features=sample, labels=label, adj_matrix=adj_matrix)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 对抗性扰乱
            fgm.attack()
            loss_adv, predict_adv = model(features=sample, labels=label, adj_matrix=adj_matrix)
            loss_adv.backward()
            fgm.restore()

            optimizer.step()

            losses += loss.item()
    
            if idx % 50 == 0:
                logging.info(f"Epoch: [{epoch+1}/{config.epochs}], Batch[{idx}/{len(train_loader)}], "
                             f"Train loss: {loss.item():.3f}")

        end_time = time.time()
        train_loss = losses / len(train_loader)
        if (epoch+1) % 5 == 0:
            logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        else:
            logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}")
        
        # 保存最后的模型
        torch.save(model.state_dict(), config.model_save_path)

    # 测试模型性能
    aiming, coverage, accuracy, absolute_true, absolute_false = TestModel(test_loader, adj_matrix, model, config.device, config.res_save_path)
    logging.info(f"precision on val {aiming:.3f}")
    logging.info(f"coverage on val {coverage:.3f}")
    logging.info(f"accuracy on val {accuracy:.3f}")
    logging.info(f"absolute_true on val {absolute_true:.3f}")
    logging.info(f"absolute_false on val {absolute_false:.3f}")
    print(f"{config.data_name}_seed{config.model_num}_lr{config.learning_rate}_E{config.epochs}_BS{config.batch_size}")
        

def TestModel(data_iter, adj_matrix, model, device, res_save_path):
    '''
    模型评估函数
    参数：
        data_iter: 数据集
        adj_matrix: 邻接矩阵
        model: 模型
        device: 设备
        res_save_path: 模型评估结果保存路径
    '''
    model.eval()
    with torch.no_grad():
        real_res = []
        pred_res = []
        for id, (x, y) in enumerate(data_iter):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            
            predict = model(features=x, adj_matrix=adj_matrix)

            y_pred = predict.sigmoid()
            y_pred = y_pred.detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            if id == 0:
                pred_res = y_pred
                real_res = label_ids
            else:
                pred_res = np.vstack((y_pred, pred_res))
                real_res = np.vstack((label_ids, real_res))

        th = threshold(pred_res, real_res)
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(pred_res > th, real_res)

        # aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(pred_res>0.5, real_res)    # pred_res>0.5 是一个布尔类型的值，True：1，False：0

        # 将结果保存到文件中
        with open(res_save_path, 'w') as f:
            f.write(f'Precision:        {aiming:.3f}\n')
            f.write(f'Coverage:         {coverage:.3f}\n')
            f.write(f'Accuracy:         {accuracy:.3f}\n')
            f.write(f'Absolute True:    {absolute_true:.3f}\n')
            f.write(f'Absolute False:   {absolute_false:.3f}\n')

        model.train()
        return aiming, coverage, accuracy, absolute_true, absolute_false


if __name__ == '__main__':
    for model_num in range(10):
        train_config = TrainConfig(model_num)
        train(train_config)
