from typing import List

import torch
import torchvision as tv
import pytorch_lightning as pl

from .oscar.modeling_bert import BertForImageCaptioning, BertConfig
from .efficientnet.net import EfficientNet


class Monocle(pl.LightningModule):

    def __init__(self, bert_config: BertConfig, ):
        super().__init__()
        
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b3',
            in_channels=3,
            include_top=False)
        self.bert = BertForImageCaptioning(bert_config)
    
    def pad_batch_roi_feat(self, sptial_embed):
        """
        sptial_embed: (number_of_box_of_batch, img_feat_dim)
        """
        pass

    def forward(self, 
            imgs: torch.Tensor,
            boxes:List[torch.Tensor],
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            masked_pos=None,
            masked_ids=None,
            is_decode=False,
            is_training=False):
        feat_map = self.backbone(imgs)
        scale = feat_map.shape[-1] / imgs.shape[-1]
        rois = tv.ops.roi_align(feat_map, boxes, 3, spatial_scale=scale, aligned=True)
        rois = rois.mean(dim=[-1, -2])
        sptial_embed = boxes
        img_feats = self.pad_batch_roi_feat(sptial_embed)

        inputs = {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'img_feats': img_feats, 
            'masked_pos': masked_pos, 'masked_ids': masked_ids,
            "is_decode": is_decode, "is_training": is_training,
        }
        return self.bert(**inputs)
    
    def training_step(self, batch, batch_idx):
        imgs, boxes, encodings = batch
        inputs = {
            'input_ids': batch[0], 'attention_mask': batch[1],
            'token_type_ids': batch[2], 'img_feats': batch[3], 
            'masked_pos': batch[4], 'masked_ids': batch[5],
            'is_decode': False, 'is_training': True,
        }
        outputs = self.forward(**inputs)
        loss, logits = outputs[:2]
        masked_ids = inputs['masked_ids']
        masked_ids = masked_ids[masked_ids != 0]
        # batch_score = compute_score_with_logits(logits, masked_ids)
        batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recons = self.decoder(z)
        reconstruction_loss = nn.functional.mse_loss(recons, x)
        self.log('val_reconstruction', reconstruction_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)