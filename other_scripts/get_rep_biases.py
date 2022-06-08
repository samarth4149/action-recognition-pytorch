import os
import sys
sys.path.append(os.path.abspath('.'))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset_config import DATASET_CONFIG
from pathlib import Path

if __name__ == '__main__':
    rep_biases = []
    for i in range(1, 11):
        file_path = next(Path(f'snapshots_hmdb51_25cls_split/tsn/split_{i}').glob('*')) / 'model_best.pth.tar'
        curr_acc = torch.load(file_path)['best_top1']
        rep_biases.append(np.log2((curr_acc.item() * DATASET_CONFIG[f'hmdb51_cls_split_{i}']['num_classes'])/100.))
    print(rep_biases[::-1])
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(21, 51, 3), rep_biases[::-1])
    # ax.set_xlabel('Num classes')
    # ax.set_ylabel('Rep bias')
    # fig.savefig('rep_biases.png', dpi=300)
    