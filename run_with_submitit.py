# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
from audioop import add
import os
import uuid
from pathlib import Path
from models.model_builder import build_model

from train import main_worker
from opts import arg_parser
# from classifier.config import ClassifierConfig
import torch
import submitit
import torch.multiprocessing as mp
from utils.dataset_config import get_dataset_config
import copy


def parse_args(get_defaults=False):
    classification_parser = arg_parser()
    parser = argparse.ArgumentParser("Submitit for Action Recognition", parents=[classification_parser], add_help=False)
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=360, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument('--long_job', default=False, action='store_true', help='Requests 48hr job')
    parser.add_argument('--email', default=False, action='store_true', help='Whether to get email notifications regarding job')
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    if get_defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def get_init_file(root):
    # Init file must not exist, but it's parent dir must exist.
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    init_file = Path(root) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.orig_args = copy.deepcopy(args)

    def __call__(self):
        self._setup_gpu_args()
        self.args.distributed = self.args.world_size > 1 or self.args.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        if self.args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            self.args.world_size = ngpus_per_node * self.args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, self.args))
        else:
            # Simply call main_worker function
            main_worker(self.args.gpu, ngpus_per_node, self.args)

    def checkpoint(self):
        import os
        import submitit
        self.args = copy.deepcopy(self.orig_args)

        self.args.dist_url = get_init_file(root='dist_init_files').as_uri()
        # _, arch_name = build_model(self.args)
        log_folder = os.path.join(self.args.logdir, self.args.arch_name)
        checkpoint_file = os.path.join(log_folder, "checkpoint.pth.tar")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        else:
            raise Exception('No checkpoint to resume from')

        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        # self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.local_rank = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main1(args):
    args.num_classes = get_dataset_config(args.dataset)[0]
    _, args.arch_name = build_model(args)
    if args.job_dir == "":
        args.job_dir = os.path.join(args.logdir, args.arch_name, 'slurm')

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    addnl_params = {
        'gres': 'gpu:{:d}'.format(num_gpus_per_node)
    }
    if args.long_job:
        addnl_params['qos'] = 'dcs-48hr'

    if args.email:
        addnl_params['mail_type'] = 'FAIL'
        addnl_params['mail_user'] = 'rpanda@ibm.com'

    executor.update_parameters(
        mem_gb=20 * num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_signal_delay_s=120,
        slurm_additional_parameters=addnl_params,
        **kwargs
    )

    executor.update_parameters(name=args.arch_name)

    args.dist_url = get_init_file(root='dist_init_files').as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

def main():
    args = parse_args()
    main1(args)

if __name__ == "__main__":
    main()