import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    rep_biases = []
    rep_biases = np.array([3.5193741990895444, 3.536052976149792, 3.726104675006119, 3.884272379908379, 3.9818527717182954, 4.058893653831521, 4.093109444162469, 4.087462912471389, 4.1154771433351, 4.191141551625815,])
    # for i in range(1, 11):
    #     curr_acc = torch.load(f'snapshots_hmdb51_cls_split/tsn/split_{i}/hmdb51_cls_split_{i}-rgb-resnet-50-ts-max-f8-bs32-lr1e-03-wd5e-04/model_best.pth.tar')['best_top1']
    #     rep_biases.append(np.log2((curr_acc.item() * DATASET_CONFIG[f'hmdb51_cls_split_{i}']['num_classes'])/100.))
    # print(rep_biases[::-1])
    fig, ax = plt.subplots()
    ax.plot(np.arange(21, 51, 3), rep_biases, marker='o')
    ax.set_xlabel('Num classes')
    ax.set_ylabel('Rep bias')
    fig.savefig('rep_biases.png', dpi=300)
    