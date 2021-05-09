import tokenizers
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
from dataset.pl_bms import LitBBMS, LitDetBBMS


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


def test_base_dataset(batch_size=8, max_cap_len=None):
    from dataset import bms_caption, collator
    global tk_model_file
    train_dir = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val"
    labels = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv"

    tokenizer = Tokenizer.from_file(tk_model_file)
    bbms_coll = collator.EncodedBatchCollator()
    bbms = bms_caption.EncodedBBMS(
        train_dir, labels, tokenizer,
        mlm=False, det_inchi=False, max_cap_len=max_cap_len, mask_prob=0.001)
    loader = DataLoader(bbms, batch_size=batch_size, collate_fn=bbms_coll, num_workers=0)
    
    print(len(bbms))
    print(bbms[1][2:])
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


def test_beam_serach_bert():

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.cuda()
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    max_length = 200
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    # bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)
    bert = BertForImageCaptioning(config)
    bert.cuda()
    # bert = torch.jit.script(bert)
    # print(bert)

    loader = test_base_dataset(max_cap_len=max_length)
    tokenizer = loader.dataset.tokenizer
    sample = next(iter(loader))
    sample = [sample[0]] + [to_cuda(s) for s in sample[1:]]
    _, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
    fake_img_feat = torch.normal(0, 1, size=[img.shape[0], atten_mask.shape[-1] - ids.shape[-1], 1540])
    fake_img_feat = fake_img_feat.cuda()

    inputs = {
        'input_ids': ids,
        'attention_mask': atten_mask,
        'token_type_ids': type_ids,
        'img_feats': fake_img_feat, 
        'masked_pos': mask_pos,
        'is_decode': True,
        
        # hyperparameters of beam search
        'max_length': max_length,
        'num_beams': 3,
        "temperature": 1.0,
        "top_k": 3,
        "top_p": 1.0,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "num_return_sequences": 1,
        "num_keep_best": 1,
        "od_labels_start_posid": -1,

        "bos_token_id": tokenizer.token_to_id("[CLS]"), 
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
        "eos_token_ids": tokenizer.token_to_id("[SEP]"), 
        "mask_token_id": tokenizer.token_to_id("[MASK]"),
    }
    bert.train()
    output = bert(**inputs)
    decoded, logprobs = output
    
    print([o.shape for o in output[:2]])


def test_monocle(ckpt=None, is_training=True, device='cuda:1', max_length=None, beam_search=False,):
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

    loader = test_base_dataset(batch_size=1, max_cap_len=max_length)
    
    for i, sample in enumerate(loader):
        # n = loader.dataset.imgids.index('4cf6b16ffa89')
        # sample = loader.dataset[n]
        # sample = EncodedBatchCollator()([sample])
        if i > 2: break
        sample = sample[1:]
        sample = (to_cuda(s) for s in sample)
        img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample

        if beam_search:
            inputs = {
                'attention_mask': atten_mask,
                'token_type_ids': type_ids,
                'masked_pos': mask_pos,
                'is_decode': True,
                
                # hyperparameters of beam search
                'max_length': max_length,
                'num_beams': 3,
                "temperature": 1.0,
                "top_k": 3,
                "top_p": 1.0,
                "repetition_penalty": 1,
                "length_penalty": 1,
                "num_return_sequences": 1,
                "num_keep_best": 1,
                "od_labels_start_posid": -1,

                "bos_token_id": tokenizer.token_to_id("[CLS]"), 
                "pad_token_id": tokenizer.token_to_id("[PAD]"),
                "eos_token_ids": tokenizer.token_to_id("[SEP]"), 
                "mask_token_id": tokenizer.token_to_id("[MASK]"),
            }
        else:
            inputs = {
                'attention_mask': atten_mask,
                'token_type_ids': type_ids,
                'masked_pos': mask_pos,
                'masked_ids': mask_ids,
                'is_decode': False,
                'is_training': is_training,
            }
        output = bert(img, boxes, ids, **inputs)
        
        print(f'[{i}] seq_len: {ids.shape}, img size: {img.shape}, img_val_min_max: {img.min()}/{img.max()}')
        if not is_training:
            if beam_search:
                decoded = output[0]
                print(decoded)
            else:
                pred_ids = torch.argmax(output[0], dim=-1)
                pred_strs = tokenizer.decode_batch(pred_ids.cpu().numpy())
                print(mask_ids)
                print(pred_ids[:, 1:])
        else:
            loss, logits = output[:2]
            print('loss: ', loss)
        print('-' * 100)


def test_lit_data():
    global tk_model_file
    tokenizer = Tokenizer.from_file(tk_model_file)
    lit_dataset = LitDetBBMS(
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train',
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val',
        tokenizer,
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv',
        num_worker=0,
        batch_size=8,
    )

    loader = lit_dataset.train_dataloader()
    for i, data in enumerate(loader):
        for b in range(len(data[0])):
            for d in data:
                print(d[b])
            if b > 2: break
        if i > 1: break


with logger.catch(reraise=True):
    # test_visual_feat_extract()
    # test_base_dataset()
    # test_img_cap_bert()
    # test_monocle(
    #     is_training=False,
    #     beam_search=False,
    #     max_length=200,
    #     ckpt="/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_3/checkpoints/epoch=0-step=53999.ckpt"
    # )
    # test_monocle()

    # test_monocle_pl_trainer(
    #     ckpt="/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_4/checkpoints/epoch=0-step=68938.ckpt",
    #     overfit=False)

    # test_beam_serach_bert()

    test_lit_data()