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
from dataset import pl_bms

from model.oscar.modeling_bert import BertForImageCaptioning
from model.efficientnet.net import EfficientNet
from model.monocle import Monocle
from dataset.pl_bms import LitBBMS


# tk_model_file = "./checkpoints/bms_worldpiece_tokenizer.json"
tk_model_file = "./checkpoints/bms_tokenizer.json"


def test_visual_feat_extract():
    backbone = EfficientNet.from_pretrained(
        'efficientnet-b3',
        in_channels=3,
        include_top=False)
    print(backbone)

    x1 = torch.ones(1, 3, 320, 320)
    feat_map = backbone(x1)
    print(feat_map.shape)

    x2 = torch.arange(0, 100).view(1, 1 ,10, 10).float()
    boxes = [
        torch.tensor([[10, 10, 40, 40]]).float()
    ]
    rois = tv.ops.roi_align(x2, boxes, output_size=4, spatial_scale=.1, aligned=True)
    print(rois)


def test_base_dataset(batch_size=8):
    from dataset import bms_caption, collator
    global tk_model_file
    train_dir = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val"
    labels = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv"

    tokenizer = Tokenizer.from_file(tk_model_file)
    bbms_coll = collator.EncodedBatchCollator()
    bbms = bms_caption.EncodedBBMS(train_dir, labels, tokenizer, mlm=False, max_cap_len=200)
    loader = DataLoader(bbms, batch_size=batch_size, collate_fn=bbms_coll, num_workers=0)
    
    print(len(bbms))
    print(bbms[1])
    for batch_ndx, sample in enumerate(loader):
        # print(sample)
        if batch_ndx > 16: break
    return loader


def test_img_cap_bert():

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.cuda()
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    # bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)
    bert = BertForImageCaptioning(config)
    bert.cuda()
    # print(bert)

    loader = test_base_dataset()
    sample = next(iter(loader))
    sample = (to_cuda(s) for s in sample)
    _, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
    fake_img_feat = torch.normal(0, 1, size=[img.shape[0], atten_mask.shape[-1] - ids.shape[-1], 1540])
    fake_img_feat = fake_img_feat.cuda()
    inputs = {
        'input_ids': ids, 'attention_mask': atten_mask,
        'token_type_ids': type_ids, 'img_feats': fake_img_feat, 
        'masked_pos': mask_pos, 'masked_ids': mask_ids,
        'is_decode': False,
    }
    bert.train()
    output = bert(**inputs)
    loss = output[0]
    loss.backward()
    print([o.shape for o in output[:2]])


def test_monocle(ckpt=None, is_training=True, device='cuda:1'):
    """
    testing burt force token-wise inference
    """

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.to(device)
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    global tk_model_file
    tokenizer = Tokenizer.from_file(tk_model_file)
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    if ckpt:
        bert = Monocle.load_from_checkpoint(ckpt, bert_config=config)
    else:
        bert = Monocle(config)
    bert.to(device)
    # print(bert)

    loader = test_base_dataset(batch_size=1)
    
    # sample = next(iter(loader))
    # sample = (to_cuda(s) for s in sample)
    
    # img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
    # inputs = {
    #     'attention_mask': atten_mask,
    #     'token_type_ids': type_ids,
    #     'masked_pos': mask_pos,
    #     'masked_ids': mask_ids,
    #     'is_decode': False,
    #     'is_training': True,
    # }
    # bert.train()
    # output = bert(img, boxes, ids, **inputs)
    # loss = output[0]
    # loss.backward()
    # print('-' * 100)
    # print(output)
    # print([o.shape for o in output[:2]])

    for i, sample in enumerate(loader):
        # n = loader.dataset.imgids.index('4cf6b16ffa89')
        # sample = loader.dataset[n]
        # sample = EncodedBatchCollator()([sample])
        if i > 1: break
        sample = sample[1:]
        sample = (to_cuda(s) for s in sample)
        img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
        inputs = {
            'attention_mask': atten_mask,
            'token_type_ids': type_ids,
            'masked_pos': mask_pos,
            'masked_ids': mask_ids,
            'is_decode': False,
            'is_training': is_training,
        }
        print(f'[{i}] seq_len: {ids.shape}, img size: {img.shape}, img_val_min_max: {img.min()}/{img.max()}')
        output = bert(img, boxes, ids, **inputs)
        if not is_training:
            pred_ids = torch.argmax(output[0], dim=-1)
            pred_strs = tokenizer.decode_batch(pred_ids.cpu().numpy())
            print(pred_ids)
        else:
            loss, logits = output[:2]
            print(loss)



def test_monocle_pl_trainer(overfit=True, ckpt=None):
    global tk_model_file
    tokenizer = Tokenizer.from_file(tk_model_file)
    lit_dataset = LitBBMS(
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train',
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val',
        tokenizer,
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv',
        num_worker=2,
        batch_size=12,
    )

    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    pretrain_dir = "./config/bms_img_cap_bert/"
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
            checkpoint_callback=True,
            callbacks=[],
            default_root_dir='checkpoints/dev',
            gpus=1,
            precision=16,
            max_steps=1000,
            overfit_batches=64,
        )
    else:
        trainer = pl.Trainer(
            fast_dev_run=False,
            accumulate_grad_batches=2,
            gradient_clip_val=1.0,
            val_check_interval=2000,
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
    # test_visual_feat_extract()
    # test_base_dataset()
    # test_img_cap_bert()
    # test_monocle(is_training=True, ckpt="/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_2/checkpoints/epoch=0-step=53365.ckpt")
    # test_monocle()
    test_monocle_pl_trainer(
        ckpt="/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_2/checkpoints/epoch=0-step=53365.ckpt",
        overfit=False)