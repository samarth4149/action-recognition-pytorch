import torch
import os
from pathlib import Path

if __name__ == '__main__':
    new_root = Path('snapshots_somethingsomething_sweep/tsn_old_pytorch')
    for path in Path('snapshots_somethingsomething_sweep/tsn').glob('*'):
        print(path)
        ckpt = torch.load(path / 'checkpoint.pth.tar')
        os.makedirs(new_root / path.name, exist_ok=True)
        torch.save(ckpt, new_root / path.name / 'checkpoint.pth.tar', _use_new_zipfile_serialization=False)