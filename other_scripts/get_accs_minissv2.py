import os
import sys
sys.path.append(os.path.abspath('.'))
from run_with_submitit import parse_args
from models.model_builder import build_model
import torch

from utils.dataset_config import get_dataset_config

def set_common():
    args = parse_args(get_defaults=True)
    args.ngpus = 8
    args.nodes = 1
    args.datadir = '/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/something2something-v2' 
    args.dataset = 'mini_st2stv2' 
    args.frames_per_group = 1 
    args.groups = 8 
    args.logdir = 'snapshots_somethingsomething_sweep/tsn' 
    args.backbone_net = 'resnet'
    args.workers = 64
    args.epochs = 100
    args.depth = 50
    args.imagenet_pretrained = False
    return args

if __name__ == '__main__':
    args = parse_args()
    
    
    run_idx = 0
    for wd in [0.0005, 0.001, 0.0001]:
        for bs in [64, 32]:
            for lr in [0.0001, 0.0005, 0.001]:
                if run_idx == 0:
                    run_idx += 1
                    continue
                
                args = set_common()
                args.weight_decay = wd
                args.lr = lr
                args.batch_size = bs
                
                args.num_classes = get_dataset_config(args.dataset)[0]
                _, arch_name = build_model(args)
                ckpt_file = os.path.join(args.logdir, arch_name, 'model_best.pth.tar')
                ckpt = torch.load(ckpt_file)
                acc = ckpt['best_top1']
                print('{:.2f}'.format(acc))