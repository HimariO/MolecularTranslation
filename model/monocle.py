import functools
from os import pread
from typing import List, Tuple

import torch
import torchvision as tv
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from .oscar.modeling_bert import BertForImageCaptioning, BertConfig
from .oscar.mlm import compute_score_with_logits
from .efficientnet.net import EfficientNet


PAD_TOKEN_ID = 3

class Monocle(pl.LightningModule):

    def __init__(self, bert_config: BertConfig, tokenizer: Tokenizer=None):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b3',
            in_channels=3,
            include_top=False)
        self.bert = BertForImageCaptioning(bert_config)
        self.tokenizer = tokenizer

        self.train_accuracy = pl.metrics.classification.Accuracy()
        self.val_accuracy = pl.metrics.classification.Accuracy()
    
    def pad_batch_roi_feat(self, raw_img_embed: torch.Tensor, boxes_list: List[torch.Tensor]) -> torch.Tensor:
        """
        raw_img_embed: (number_of_box_of_batch, img_feat_dim)
        """
        max_num = max([len(b) for b in boxes_list])
        batch_size = len(boxes_list);
        # batched_feat = torch.zeros([batch_size, max_num, raw_img_embed.shape[1]])
        padded = []
        curr_idx = 0
        for boxes in boxes_list:
            sample_embed = raw_img_embed[curr_idx: curr_idx + len(boxes)]
            padded_embed = F.pad(sample_embed, [0, 0, 0, max_num - len(boxes)], value=0.0)
            padded.append(padded_embed)
            curr_idx += len(boxes)
        return torch.stack(padded, dim=0)

    def forward(self, 
            imgs: torch.Tensor,
            boxes: List[torch.Tensor],
            input_ids: torch.Tensor,
            attention_mask=None,
            token_type_ids=None,
            masked_pos=None,
            masked_ids=None,
            is_decode=False,
            is_training=False) -> Tuple[torch.Tensor]:
        feat_map = self.backbone(imgs)
        scale = feat_map.shape[-1] / imgs.shape[-1]
        rois = tv.ops.roi_align(feat_map, boxes, 3, spatial_scale=scale, aligned=True)
        rois = rois.mean(dim=[-1, -2])
        
        w = imgs.shape[-1]
        h = imgs.shape[-2]
        img_dim = torch.tensor([[w, h, w, h]], dtype=torch.float32)
        img_dim = img_dim.to(imgs.device)
        sptial_embed = torch.cat([imgbox / img_dim for imgbox in boxes], dim=0)

        raw_img_feat = torch.cat([rois, sptial_embed], dim=1)
        img_feats = self.pad_batch_roi_feat(raw_img_feat, boxes)
        assert img_feats.shape[1] + input_ids.shape[1] == attention_mask.shape[1], \
            f"{img_feats.shape[1]} + {input_ids.shape[1]} != {attention_mask.shape[1]}"

        inputs = {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'img_feats': img_feats, 
            'masked_pos': masked_pos, 'masked_ids': masked_ids,
            "is_decode": is_decode, "is_training": is_training,
        }
        output = self.bert(**inputs)
        if is_training:
            logits = output[1]
            if torch.isnan(logits).any():
                import pdb; pdb.set_trace()
        return output
    
    def training_step(self, batch, batch_idx):
        keys, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = batch
        inputs = {
            'attention_mask': atten_mask,
            'token_type_ids': type_ids,
            'masked_pos': mask_pos,
            'masked_ids': mask_ids,
            'is_decode': False,
            'is_training': True,
        }
        outputs = self.forward(img, boxes, ids, **inputs)
        loss, logits = outputs[:2]
        pred = F.softmax(logits, dim=-1)

        masked_ids = mask_ids[mask_ids != 0]
        batch_score = compute_score_with_logits(logits, masked_ids)
        batch_acc = torch.sum(batch_score.float()) / torch.sum(mask_pos)

        self.log('train_loss', loss)
        self.log('train_acc_step', self.train_accuracy(pred, masked_ids), prog_bar=True)
        # self.log('train_acc_step', batch_acc, prog_bar=True)
        if batch_idx % 100 == 0:
            pred_ids = pred.argmax(dim=-1)
            tensorboard = self.logger.experiment
            tensorboard.add_histogram('train_batch_pred', pred_ids, global_step=batch_idx)
        
        # try:
        #     self.train_accuracy(pred, masked_ids)
        # except:
        #     import pdb; pdb.set_trace()
        # if batch_acc > 0.999:
        #     import pdb; pdb.set_trace()
        return loss

    def validation_step(self, batch, batch_idx):
        keys, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = batch
        inputs = {
            'attention_mask': atten_mask,
            'token_type_ids': type_ids,
            'masked_pos': mask_pos,
            'masked_ids': mask_ids,
            'is_decode': False,
            'is_training': False,
        }
        outputs = self.forward(img, boxes, ids, **inputs)
        logits = outputs[0]
        pred = F.softmax(logits, dim=-1)

        masked_ids = mask_ids[mask_ids != 0]
        # batch_score = compute_score_with_logits(logits, masked_ids)
        # batch_acc = torch.sum(batch_score.float()) / torch.sum(mask_pos)
        if not self.tokenizer is None and batch_idx % 50 == 0:
            pred_ids = pred.argmax(dim=-1)
            pred_ids = pred_ids.cpu().numpy().tolist()
            pred_ids = [
                [p for p in pred if p != PAD_TOKEN_ID]
                for pred in pred_ids
            ]
            pred_strs = self.tokenizer.decode_batch(pred_ids)
            log_str = "\n\n".join([f"[{j}] {s}" for j, s in enumerate(pred_strs)])

            tensorboard = self.logger.experiment
            tensorboard.add_text('train_batch_pred_txt', log_str, global_step=batch_idx)

        self.val_accuracy(pred[mask_pos != 0], masked_ids)
        return {
            'predict': torch.max(logits, -1)[1].data
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_aucc_epoch', self.val_accuracy.compute(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)