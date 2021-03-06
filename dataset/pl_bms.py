import os
import random

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from .bms_caption import EncodedBBMS, BinedBatchSampler
from .collator import EncodedBatchCollator


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class RandSampleDataset:

    def __init__(self, datasets) -> None:
        self.datasets = datasets
        self.min_size = min([len(d) for d in datasets])
    
    def __getitem__(self, index):
        index = index % self.min_size
        return random.choice(self.datasets)[index]
    
    def __len__(self,):
        return self.min_size


class LitBBMS(pl.LightningDataModule):

    def __init__(self, train_dir: str, val_dir: str, tokenizer: Tokenizer, anno_csv: str,
                val_anno_csv=None, test_dir=None, batch_size=8, num_worker=4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.anno_csv = anno_csv
        self.val_anno_csv = anno_csv if val_anno_csv is None else val_anno_csv
    
    def train_dataloader(self) -> EncodedBBMS:
        mlm_dataset = EncodedBBMS(
            self.train_dir,
            self.anno_csv,
            self.tokenizer,
            mlm=True)
        # batch_sampler = BinedBatchSampler(mlm_dataset, batch_size=self.batch_size)
        # loader = DataLoader(
        #     mlm_dataset,
        #     num_workers=self.num_worker,
        #     batch_sampler=batch_sampler,
        #     collate_fn=EncodedBatchCollator(),
        #     worker_init_fn=worker_init_fn)
        
        # mask_dataset = EncodedBBMS(
        #     self.train_dir,
        #     self.anno_csv,
        #     self.tokenizer,
        #     mlm=False)
        zip_dataset = RandSampleDataset([mlm_dataset, ])
        loader = DataLoader(
            zip_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator(),
            worker_init_fn=worker_init_fn)
        return loader
    
    def val_dataloader(self) -> EncodedBBMS:
        dataset = EncodedBBMS(
            self.val_dir,
            self.val_anno_csv,
            self.tokenizer,
            mlm=True)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator(),
            worker_init_fn=worker_init_fn)
        return loader


class LitDetBBMS(LitBBMS):

    def train_dataloader(self) -> EncodedBBMS:
        """
        Task1: inchi str from detection atom/bond graph with added mask and token swap
        """
        # mlm_dataset = EncodedBBMS(
        #     self.train_dir,
        #     self.anno_csv,
        #     self.tokenizer,
        #     mlm=True,
        #     det_inchi=True,)
        """
        Task2: completly masked input token to target inchi str
        """
        # mask_dataset = EncodedBBMS(
        #     self.train_dir,
        #     self.anno_csv,
        #     self.tokenizer,
        #     mlm=False)
        """
        Task3: translate inaccurate inchi str from detection atom/bond graph to target inchi str
        """
        translate_dataset = EncodedBBMS(
            self.train_dir,
            self.anno_csv,
            self.tokenizer,
            mlm=False,
            full_atten=True,
            det_inchi=True,)
        zip_dataset = RandSampleDataset([translate_dataset])
        loader = DataLoader(
            zip_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator(),
            worker_init_fn=worker_init_fn)
        return loader
    
    def val_dataloader(self) -> EncodedBBMS:
        dataset = EncodedBBMS(
            self.val_dir,
            self.val_anno_csv,
            self.tokenizer,
            mlm=False,
            det_inchi=True,
            mask_prob=0.0)
        loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator(),
            worker_init_fn=worker_init_fn)
        return loader