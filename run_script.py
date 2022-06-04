from run_with_submitit import main1, parse_args

if __name__ == '__main__':
    args = parse_args()
    args.ngpus = 8
    args.nodes = 1
    args.datadir = '/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/something2something-v2' 
    args.dataset = 'mini_st2stv2' 
    args.frames_per_group = 1 
    args.groups = 8 
    args.logdir = 'snapshots_somethingsomething_sweep/' 
    args.backbone_net = 'resnet'
    args.workers = 64
    args.epochs = 100
    args.depth = 50
    args.imagenet_pretrained = False
    
    run_idx = 0
    for wd in [0.0005, 0.001, 0.0001]:
        for bs in [64, 32]:
            for lr in [0.0001, 0.0005, 0.001]:
                if run_idx == 0:
                    run_idx += 1
                    continue
                args.weight_decay = wd
                args.lr = lr
                args.batch_size = bs
                main1(args)
                run_idx += 1