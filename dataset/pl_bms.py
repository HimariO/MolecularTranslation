import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from .bms_caption import EncodedBBMS
from .collator import EncodedBatchCollator


class LitBBMS(pl.LightningDataModule):

    def __init__(self, train_dir: str, val_dir: str, tokenizer: Tokenizer, anno_csv: str,
                val_anno_csv=None, batch_size=8, num_worker=4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.anno_csv = anno_csv
        self.val_anno_csv = anno_csv if val_anno_csv is None else val_anno_csv
    
    def train_dataloader(self) -> EncodedBBMS:
        dataset = EncodedBBMS(
            self.train_dir,
            self.anno_csv,
            self.tokenizer,
            mlm=True)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator())
        return loader
    
    def val_dataloader(self) -> EncodedBBMS:
        dataset = EncodedBBMS(
            self.val_dir,
            self.val_anno_csv,
            self.tokenizer,
            mlm=False)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            collate_fn=EncodedBatchCollator())
        return loader