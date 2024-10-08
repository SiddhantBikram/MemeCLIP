import argparse
import random
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import Custom_Collator, load_dataset
from MemeCLIP import create_model, MemeCLIP
from configs import cfg
import os
import torchmetrics
from tqdm import tqdm

torch.use_deterministic_algorithms(False)

def main(cfg):

    seed_everything(cfg.seed, workers=True)

    dataset_train = load_dataset(cfg=cfg, split='train')
    dataset_val = load_dataset(cfg=cfg, split='val')
    dataset_test = load_dataset(cfg=cfg, split='test')

    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))
    print("Number of test examples:", len(dataset_test))


    collator = Custom_Collator(cfg)

    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=collator, num_workers=0)
    val_loader = DataLoader(dataset_test, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size,
                                 collate_fn=collator, num_workers=0)
    
    model = create_model(cfg)

    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}

    monitor = "val/auroc"
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.checkpoint_path, filename='model',
                                          monitor=monitor, mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)


    trainer = Trainer(accelerator='gpu', devices=cfg.gpus, max_epochs=cfg.max_epochs, callbacks=[checkpoint_callback], deterministic=False)

    if cfg.reproduce == False:

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    model = MemeCLIP.load_from_checkpoint(checkpoint_path = cfg.checkpoint_file, cfg = cfg) 
    trainer.test(model, dataloaders=test_loader)

if __name__ == '__main__':
      main(cfg)

