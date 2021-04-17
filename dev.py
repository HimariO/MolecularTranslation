import torch
import torchvision as tv
from torch.utils.data import DataLoader
from loguru import logger
from model.efficientnet.net import EfficientNet
from tokenizers import Tokenizer


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
        break


with logger.catch():
    test_base_dataset()