import os
from deepISA.modeling.trainer import Trainer
import numpy as np
import json
import torch


class MemmapData:
    """Minimal container for memmapped genomic data."""
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        self.X = np.memmap(os.path.join(data_dir, "X.npy"), dtype='float32', mode='r', shape=tuple(meta['X']))
        self.Yr = np.memmap(os.path.join(data_dir, "Yr.npy"), dtype='float32', mode='r', shape=tuple(meta['Yr']))
        self.Yc = np.memmap(os.path.join(data_dir, "Yc.npy"), dtype='float32', mode='r', shape=tuple(meta['Yc']))
        self.n_samples = self.X.shape[0]

    def __len__(self):
        return self.n_samples



def train_model(model,
                device,
                train_dat_dir,
                model_dir,
                trainer_config=None,
                mode='dual'):
    
    if trainer_config is None:
        trainer_config={
            "epochs": 10,
            "batch_size": 128,
            "patience": 3,
            "min_delta": 0.001, # minimum change in the monitored metric to qualify as an improvement
            "learning_rate": 1e-3,
            "save_one_fourth": False,
            "save_one": False
        }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, "trainer_config.json"), 'w') as f:
        json.dump(trainer_config, f, indent=4)
    
    # 2. Create Datasets
    train_ds = MemmapData(os.path.join(train_dat_dir, "train"))
    val_ds = MemmapData(os.path.join(train_dat_dir, "val"))
    test_ds  = MemmapData(os.path.join(train_dat_dir, "test"))
    
    # 5. Training Execution
    trainer = Trainer(
        model=model,
        mode=mode,
        train_dat=train_ds,
        val_dat=val_ds,
        test_dat=test_ds,
        device =device,
        model_dir = model_dir,
        trainer_config = trainer_config
    )
    
    trainer.train()
    # Clean up GPU memory immediately
    del trainer
    del model
    torch.cuda.empty_cache()
    return None
    