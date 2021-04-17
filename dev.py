import torch
import torchvision as tv
from torch.utils.data import DataLoader
from loguru import logger
from tokenizers import Tokenizer
from transformers.models.bert.modeling_bert import BertConfig

from model.oscar.modeling_bert import BertForImageCaptioning
from model.efficientnet.net import EfficientNet

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


def test_base_dataset():
    from dataset import bms_caption, collator
    train_dir = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train"
    labels = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv"

    tk_model_file = "./checkpoints/bms_tokenizer.json"
    tokenizer = Tokenizer.from_file(tk_model_file)
    bbms_coll = collator.EncodedBatchCollator()
    bbms = bms_caption.EncodedBBMS(train_dir, labels, tokenizer, mlm=True)
    loader = DataLoader(bbms, batch_size=8, collate_fn=bbms_coll, num_workers=0)
    
    print(len(bbms))
    print(bbms[1])
    for batch_ndx, sample in enumerate(loader):
        print(sample)
        return sample


def test_img_cap_bert():
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    # bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)
    bert = BertForImageCaptioning(config)
    print(bert)

    sample = test_base_dataset()
    img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
    fake_img_feat = torch.normal(0, 1, size=[img.shape[0], atten_mask.shape[-1] - ids.shape[-1], 1540])
    inputs = {
        'input_ids': ids, 'attention_mask': atten_mask,
        'token_type_ids': type_ids, 'img_feats': fake_img_feat, 
        'masked_pos': mask_pos, 'masked_ids': mask_ids
    }
    bert.train()
    output = bert(**inputs)
    print([o.shape for o in output])


with logger.catch(reraise=True):
    # test_visual_feat_extract()
    # test_base_dataset()
    test_img_cap_bert()