import torch
import torch.nn as nn


def wta_enhancement(input):
    for i in range(input.size(0)):  # 对每个批次样本进行处理
        # 找到最大值的索引
        max_index = torch.argmax(input[i])
        # 获取最大值
        max_value = input[i, max_index]
        # 增强处理（乘以2并平方）
        enhanced_value = (2 * max_value) ** 2
        # 更新输出向量
        input[i, max_index] = enhanced_value
    return input
