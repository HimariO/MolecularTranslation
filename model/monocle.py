import torch
import torchvision as tv
import pytorch_lightning as pl

from .oscar.modeling_bert import BertForImageCaptioning
from .efficientnet.net import EfficientNet


class Monocle(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b3',
            in_channels=3,
            include_top=False)
        self.bert = BertForImageCaptioning

    def forward(self, imgs, boxes):
        tv.ops.roi_align()