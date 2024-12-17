import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import Accuracy


def threshold(predict, labels):
    '''
    动态阈值
    '''
    best_acc = 0
    best_threshold = 0

    for threshold in range(30, 70):
        threshold = threshold / 100
        acc = Accuracy(predict > threshold, labels)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold
