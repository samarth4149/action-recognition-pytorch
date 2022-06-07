from email import parser
import numpy as np
import torch
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Getting accs')
    parser.add_argument('--snapshot_dir', type=str, default='snapshots_hmdb51_cls_split/tsn_kinetics_pt')
    args = parser.parse_args()
    
    accs = []
    for i in range(1, 11):
        model_path = next((Path(args.snapshot_dir) / f'split_{i}').glob('*')) / 'model_best.pth.tar'
        curr_model = torch.load(model_path)
        accs.append(curr_model['best_top1'].item())
        
    print(accs[::-1])