import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_config import DATASET_CONFIG

if __name__ == '__main__':
    rep_biases = []
    for i in range(1, 11):
        curr_acc = torch.load(f'snapshots_hmdb51_cls_split/tsn/split_{i}/hmdb51_cls_split_{i}-rgb-resnet-50-ts-max-f8-bs32-lr1e-03-wd5e-04/model_best.pth.tar')['best_top1']
        rep_biases.append(np.log((curr_acc.item() * DATASET_CONFIG[f'hmdb51_cls_split_{i}']['num_classes'])/100.))
    fig, ax = plt.subplots()
    ax.plot(np.arange(21, 51, 3), rep_biases[::-1])
    ax.set_xlabel('Num classes')
    ax.set_ylabel('Rep bias')
    fig.savefig('rep_biases.png', dpi=300)
    