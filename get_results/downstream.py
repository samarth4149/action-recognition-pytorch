from run_with_submitit import main1, parse_args
from models.model_builder import build_model
import os

from utils.dataset_config import get_dataset_config

from pathlib import Path

def set_common():
    args = parse_args(get_defaults=True)
    args.ngpus = 4
    args.nodes = 2
    # args.datadir = '/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/something2something-v2' 
    # args.dataset = 'mini_st2stv2' 
    args.frames_per_group = 1 
    args.groups = 8 
    # args.logdir = 'snapshots_somethingsomething_sweep/tsn' 
    # args.pretrained = '/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/synapt/moments_models/tsn_moments_model_best.pth.tar'
    # args.lin_probe = True
    # args.backbone_net = 'resnet'
    args.workers = 64
    args.epochs = 100
    args.depth = 50
    args.imagenet_pretrained = False
    return args

if __name__ == '__main__':
    args = parse_args()
    
    data_base = Path('/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/')
    
    run_idx = 0
    for model in ['tsn', 'i3d', 's3d']:
        if args.model == 'tsn':
            args.backbone_net = 'resnet'
            hyperparams = {
                'ucf101' : [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'hmdb51' : [(32, 0.001, 0.0001), (64, 0.001, 0.0001)],
                'mini_st2stv2': [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'diving48' : [(32, 0.001, 0.001), (32, 0.001, 0.0001)],
                'ikea_furniture' : [(32, 0.001, 0.001), (32, 0.0005, 0.0005)],
                'uav_human' : [(32, 0.001, 0.001), (32, 0.001, 0.0001)],
            }
        elif args.model == 'i3d':
            args.backbone_net = 'i3d_resnet'
            hyperparams = {
                'ucf101' : [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'hmdb51' : [(32, 0.001, 0.0001), (32, 0.0005, 0.0001)],
                'mini_st2stv2': [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'diving48' : [(32, 0.001, 0.001), (32, 0.001, 0.0005)],
                'ikea_furniture' : [(64, 0.0005, 0.001), (32, 0.001, 0.0005)],
                'uav_human' : [(64, 0.001, 0.0001), (64, 0.0005, 0.0001)],
            }
        else:
            args.backbone_net = 's3d_resnet'
            hyperparams = {
                'ucf101' : [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'hmdb51' : [(32, 0.001, 0.0001), (64, 0.001, 0.0001)],
                'mini_st2stv2': [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
                'diving48' : [(32, 0.001, 0.001), (32, 0.001, 0.0001)],
                'ikea_furniture' : [(64, 0.001, 0.0005), (32, 0.0005, 0.0005)],
                'uav_human' : [(32, 0.001, 0.0005), (32, 0.001, 0.0001)],
            }
        for dataset in ['ucf101', 'hmdb51', 'diving48']:
            data_dir = data_base / dataset if dataset != 'mini_st2stv2' else data_base / 'something2something-v2'
            best_acc = 0
            best_hps = None
            for (bs, lr, wd) in hyperparams[dataset]:   
                args = set_common()
                args.datadir = data_dir
                args.dataset = dataset
                if dataset in ['diving48', 'ikea_furniture', 'uav_human']:
                    args.lin_probe = False
                    args.logdir = f'expts/tsn_resnet_moments_pt/{dataset}_finetune'
                else:
                    args.lin_probe = True
                    args.logdir = f'expts/tsn_resnet_moments_pt/{dataset}_lin_probe'
                
                args.weight_decay = wd
                args.lr = lr
                args.batch_size = bs
                
                args.num_classes = get_dataset_config(args.dataset)[0]
                _, arch_name = build_model(args)
                # ckpt_file = os.path.join(args.logdir, arch_name, 'checkpoint.pth.tar')
                log_file = os.path.join(args.logdir, arch_name, 'log.log')
                with open(log_file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines[::-1]:
                        if line.startswith('Val'):
                            break
                    if '[100/100]' not in line:
                        print(f'tsn {dataset} {bs} {lr} {wd} not finished')
                    acc = float(line[line.find('Top@1: ') + len('Top@1: '):line.find('Top@5: ')])
                
                    if acc > best_acc:
                        best_acc = acc
                        best_hps = (bs, lr, wd)
                
                run_idx += 1
                
            print(f'{model} {dataset} best acc : {best_acc:.2f}')