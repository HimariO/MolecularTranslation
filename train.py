import tokenizers
from dataset import tokenizer
from dataset.collator import EncodedBatchCollator
from json import load
from numpy.lib.function_base import iterable
import torch
import torchvision as tv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from loguru import logger
from tokenizers import Tokenizer
from transformers.models.bert.modeling_bert import BertConfig

from model.oscar.modeling_bert import BertForImageCaptioning
from model.efficientnet.net import EfficientNet
from model.monocle import Monocle
from dataset.pl_bms import LitBBMS, LitDetBBMS
from dataset.tokenizer import basic_tokenizer


tk_model_file = "./checkpoints/bms_tokenizer.json"


def test_monocle_pl_trainer(overfit=True, ckpt=None):
    global tk_model_file
    # tokenizer = Tokenizer.from_file(tk_model_file)
    tokenizer = basic_tokenizer()
    lit_dataset = LitBBMS(
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train',
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val',
        tokenizer,
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv',
        num_worker=2,
        batch_size=12,
    )

    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    # pretrain_dir = "./config/bms_img_cap_bert/"
    pretrain_dir = "./config/basic_tk_bms_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    if ckpt:
        bert = Monocle.load_from_checkpoint(ckpt, bert_config=config)
    else:
        bert = Monocle(config)
    
    if overfit:
        trainer = pl.Trainer(
            # fast_dev_run=True,
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=False,
            callbacks=[],
            default_root_dir='checkpoints/overfit',
            gpus=1,
            precision=16,
            max_steps=1000,
            overfit_batches=8,
        )
    else:
        trainer = pl.Trainer(
            fast_dev_run=False,
            accumulate_grad_batches=1,
            gradient_clip_val=1.0,
            val_check_interval=2500,
            checkpoint_callback=True,
            callbacks=[],
            default_root_dir='checkpoints/dev',
            gpus=1,
            # num_nodes=2,
            # distributed_backend='ddp',
            precision=16,
            # max_steps=1000,
            # resume_from_checkpoint=resume_ckpt,
            # num_sanity_val_steps=0,
            # overfit_batches=32,
            max_epochs=100,
            # limit_train_batches=5000,
        )
    trainer.fit(bert, datamodule=lit_dataset)


with logger.catch(reraise=True):
    test_monocle_pl_trainer(
        ckpt=None,
        overfit=False)