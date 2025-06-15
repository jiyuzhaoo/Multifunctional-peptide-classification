import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from GAT import GATModel
from GAT_train import TrainConfig
from utils.threshold import threshold
from utils.evaluation import evaluate
from torch.utils.data import DataLoader
from utils.metrics import binary_class_metrics
from utils.data_helpers import MultiLabelDataset


def get_model_data(model_num):
    # 加载模型及参数
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_num in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        model_path = project_dir + f"/cache/models/model_ESM_{model_num}.bin"
    elif model_num in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
        model_path = project_dir + f"/cache/models/model_PortT5_{model_num}.bin"
    else:
        model_path = project_dir + f"/cache/models/model_RoBERTa_{model_num}.bin"

    model_config = TrainConfig(model_num)
    model = GATModel(model_config)

    if os.path.exists(model_path):
        loaded_paras = torch.load(model_path)
        model.load_state_dict(loaded_paras)
        print(f"## Successfully loaded {model_path} model for inference ......")
    else:
        print("Model not found")

    # 加载数据集
    dataset_test = MultiLabelDataset(np.load(model_config.test_feature_path), np.load(model_config.test_label_path))
    test_loader = DataLoader(dataset_test, batch_size=model_config.batch_size, shuffle=False)

    # 构建全连接图，边权重为1 (邻接矩阵)
    graph_info = np.ones((model_config.num_labels, model_config.num_labels))
    diagnoal = np.diag([1 for i in range(model_config.num_labels)])      # 创建对角矩阵
    graph_info = graph_info - diagnoal
    adj_matrix = torch.from_numpy(graph_info).float().to(model_config.device)
    
    return model, test_loader, adj_matrix


def predict(dataset, model, adj_matrix, device):
    '''
    模型预测，返回预测结果
    :para dataset 数据集
    :para model 模型
    :para adj_matrix 邻接矩阵
    :para device 设备
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
            pred_all = []
            label_all = []
            for id, (feature, label) in enumerate(dataset):
                feature = feature.to(device, dtype=torch.float32)
                label = label.to(device)
                
                predicts = model(features=feature, adj_matrix=adj_matrix)

                pred = predicts.sigmoid()
                pred = pred.detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()

                if id == 0:
                    pred_all = pred
                    label_all = label_ids
                else:
                    pred_all = np.vstack((pred_all, pred))
                    label_all = np.vstack((label_all, label_ids))
                
    return pred_all, label_all


if __name__ == '__main__':
    label_list = ['Acetylation', 'Sumoylation', 'Methylation', 'Crotonylation', 'Glycation','2-Hydroxyisobutyrylation','Malonylation','Succinylation','β-Hydroxybutyrylation']
    id2label = {i: label for i, label in enumerate(label_list)}
    sum_pred = []
    real_label = []
    start = 0
    end = 30
    step = 1
    num_model = 30

    for model_num in range(start, end, step):
        model, test_data, adj_matrix = get_model_data(model_num)
        pred, label = predict(test_data, model, adj_matrix, device='cuda')
        
        real_label = label
        if model_num == 0:
            sum_pred = pred
        else:
            sum_pred += pred
    
    pred_res = sum_pred / num_model
    th = threshold(pred_res, real_label)
    binary_class_metrics(pred_res, pred_res>th, real_label, id2label, save_path="model_GAT/result/result.txt", curve_save_path="model_GAT/parameters")
    # aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(pred_res > th, real_label)

    # # 直接在代码的最后添加写入文件的操作
    # with open("cache/result/results_fusion.txt", 'a') as file:
    #     file.write(f"Aiming:            {aiming:.3f}\n")
    #     file.write(f"Coverage:          {coverage:.3f}\n")
    #     file.write(f"Accuracy:          {accuracy:.3f}\n")
    #     file.write(f"Absolute True:     {absolute_true:.3f}\n")
    #     file.write(f"Absolute False:    {absolute_false:.3f}\n")
    #     file.write("=====================================\n")
