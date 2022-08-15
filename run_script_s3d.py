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
    # args.logdir = 'snapshots_somethingsomething_sweep/s3d_resnet' 
    args.pretrained = '/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/synapt/moments_models/i3d_moments_model_best.pth.tar'
    args.lin_probe = True
    args.backbone_net = 's3d_resnet'
    args.multiprocessing_distributed = True
    args.threed_data = True
    args.workers = 64
    args.epochs = 100
    args.depth = 50
    args.imagenet_pretrained = False
    return args

if __name__ == '__main__':
    args = parse_args()
    
    data_base = Path('/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/')
    hyperparams = {
        'ucf101' : [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
        'hmdb51' : [(32, 0.001, 0.0001), (64, 0.001, 0.0001)],
        'mini_st2stv2': [(32, 0.001, 0.0001), (32, 0.001, 0.001)],
    }
    
    run_idx = 0
    for dataset in ['mini_st2stv2', 'hmdb51', 'ucf101']:
        data_dir = data_base / dataset if dataset != 'mini_st2stv2' else data_base / 'something2something-v2'
        for (bs, lr, wd) in hyperparams[dataset]: 
            args = set_common()
            args.datadir = data_dir
            args.dataset = dataset
            args.log_dir = f'expts/s3d_resnet_moments_pt/{dataset}_lin_probe'
            args.weight_decay = wd
            args.lr = lr
            args.batch_size = bs
            
            args.num_classes = get_dataset_config(args.dataset)[0]
            _, arch_name = build_model(args)
            ckpt_file = os.path.join(args.logdir, arch_name, 'checkpoint.pth.tar')
            if os.path.exists(ckpt_file):
                print(f'Resuming from {ckpt_file}')
                args.resume = ckpt_file
            
            main1(args)
            run_idx += 1