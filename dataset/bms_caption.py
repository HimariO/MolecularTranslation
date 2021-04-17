import os
import glob
import json
import random
import pickle
from collections import namedtuple
from typing import Tuple, Union

import cv2
import torch
import numpy as np
import pandas as pd
from loguru import logger
from tokenizers import Tokenizer, Encoding
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader


MaskedEncoding = namedtuple(
    "MaskedEncoding",
    ["ids",
    "type_ids",
    "tokens",
    "offsets",
    "attention_mask",
    "masked_pos",
    "masked_ids"])

"""
Encoding token id with extended attention mask that accounting image feaetures,
len(attention_mask) == len(tokens) + len(bboxes) == len(token_ids) + len(img_feats)
"""
VisAttenEncoding = namedtuple(
    "MaskedEncoding",
    ["ids",
    "type_ids",
    "tokens",
    "offsets",
    "token_attention_mask",
    "img_attention_mask",
    "masked_pos",
    "masked_ids",])

class BoxedBMS(Dataset):

    TMP_DIR = os.path.join(os.environ['HOME'], 'bms_tmp')

    def __init__(self, dataset_dir, anno_csv) -> None:
        self.dataset_dir = dataset_dir
        self.anno_csv = anno_csv
        self.id2imgdet, self.id2cap = self.get_samples()
        self.imgids = list(self.id2imgdet.keys())
    
    def get_samples(self):
        logger.info(f"[{self.__class__.__name__}] get_samples")
        pat = os.path.join(self.dataset_dir, '**', '*.png')
        img_list = glob.glob(pat, recursive=True)
        logger.info(f"[{self.__class__.__name__}] find {len(img_list)} imgs")
        # det_json_list = [ip.replace(".png", ".json") for ip in img_list]
        
        os.makedirs(self.TMP_DIR, exist_ok=True)
        cache_file = os.path.join(self.TMP_DIR, f"{len(img_list)}.pickle")
        if os.path.exists(cache_file):
            logger.info(f'Load cached dataset samples from: {cache_file}')
            with open(cache_file, mode='rb') as f:
                id2imgdet, id2cap = pickle.load(f)
        else:
            id2imgdet = {
                os.path.basename(img).replace('.png', ''): (
                    img, img.replace(".png", ".json"))
                for img in img_list
            }
            anno_pd = pd.read_csv(self.anno_csv)
            id2cap = {row.image_id: row.InChI for _, row in anno_pd.iterrows()}

            for old_cache in glob.glob(os.path.join(self.TMP_DIR, '*.pickle')):
                logger.warning(f'Remove old samples cache: {old_cache}')
                os.remove(old_cache)
            logger.info(f'Cache dataset samples to: {cache_file}')
            with open(cache_file, mode='wb') as f:
                pickle.dump((id2imgdet, id2cap), f)
        # assert len(id2imgdet) == len(id2cap)
        return id2imgdet, id2cap
    
    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np.transpose(img, [2, 0, 1])
        return torch.tensor(img)

    def load_bbox(self, json_path, expand_ratio=1.5):
        with open(json_path, mode='r') as f:
            anno = json.load(f)
        h = float(anno['image_height'])
        w = float(anno['image_width'])
        boxes_xywh = torch.tensor([b['bbox'] for b in anno['boxes']])
        boxes_x2y2 = boxes_xywh[:, :2] + boxes_xywh[:, 2:]
        boxes = torch.cat([boxes_xywh[:, :2], boxes_x2y2], dim=1)
        
        if expand_ratio > 1.0:
            boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
            boxes_wh = boxes_xywh[:, 2:] * expand_ratio
            boxes = torch.cat([boxes_center - boxes_wh / 2, boxes_center + boxes_wh / 2], dim=1)
            
            wh = torch.tensor([[w, h, w, h]], dtype=torch.float32)
            boxes /= wh
            boxes = torch.clip(boxes, 0, 1)
            boxes *= wh
        return boxes
    
    def __len__(self):
        return len(self.id2imgdet)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, str]:
        key = self.imgids[i]
        caption = self.id2cap[key]
        img_path, bbox_json_path = self.id2imgdet[key]
        img = self.load_image(img_path)
        boxes = self.load_bbox(bbox_json_path)
        return img, boxes, caption


class EncodedBBMS(BoxedBMS):
    
    max_masked_tokens = 16

    def __init__(self, dataset_dir: str, anno_csv: str, tokenizer: Tokenizer,
                mask_prob=0.2, mlm=True,):
        super().__init__(dataset_dir, anno_csv)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mlm = mlm
    
    def random_mask_caption(self, encoding: Encoding) -> MaskedEncoding:
        tokens = encoding.tokens
        seq_a_len = len(tokens)
        masked_pos = torch.zeros(seq_a_len, dtype=torch.int)
        # randomly mask words for prediction, ignore [CLS]
        candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
        random.shuffle(candidate_masked_idx)
        num_masked = max(round(self.mask_prob * seq_a_len), 1)
        num_masked = min(num_masked, self.max_masked_tokens)
        num_masked = int(num_masked)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        masked_token = [tokens[i] for i in masked_idx]
        for pos in masked_idx:
            if random.random() <= 0.8:
                # 80% chance to be a ['MASK'] token
                tokens[pos] = self.tokenizer.token_to_id('[MASK]')
            elif random.random() <= 0.5:
                # 10% chance to be a random word ((1-0.8)*0.5)
                from random import randint
                i = randint(0, self.tokenizer.get_vocab_size())
                # self.tokenizer._convert_id_to_token(i)
                tokens[pos] = self.tokenizer.id_to_token(i)
            else:
                # 10% chance to remain the same (1-0.8-0.1)
                pass

        masked_pos[masked_idx] = 1 
        # pad masked tokens to the same length
        if num_masked < self.max_masked_tokens:
            masked_token = masked_token + (['[PAD]'] * (self.max_masked_tokens - num_masked))
        masked_ids = [self.tokenizer.token_to_id(t) for t in masked_token]  # shape: (num_masked,)

        return MaskedEncoding(
            ids=encoding.ids,
            tokens=encoding.tokens,
            type_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            offsets=encoding.offsets,
            masked_pos=masked_pos,
            masked_ids=masked_ids,
        )
    
    def extend_visual_atten(
            self,
            encoding: Union[Encoding, MaskedEncoding],
            boxes) -> VisAttenEncoding:
        atten_1d = encoding.attention_mask
        """
        [CLS][CAPTION][SEP][BOX_FEAT][SEP]
        | atten_1d         | extened part|
        """
        a = len(atten_1d)
        b = len(boxes) + 1
        n = a + b
        attention_mask = torch.zeros([n, n], dtype=torch.long)
        # import pdb; pdb.set_trace()
        attention_mask[:a, :a] = torch.tril(torch.ones((a, a), dtype=torch.long))
        attention_mask[a:, a:] = 1
        attention_mask[:a, a:] = 1
        
        return VisAttenEncoding(
            ids=torch.tensor(encoding.ids),
            tokens=encoding.tokens,
            type_ids=torch.tensor(encoding.type_ids),
            token_attention_mask=attention_mask[:a],
            img_attention_mask=attention_mask[a:],
            offsets=encoding.offsets,
            masked_ids=torch.tensor(encoding.masked_ids) if hasattr(encoding, 'masked_ids') else None,
            masked_pos=torch.tensor(encoding.masked_pos) if hasattr(encoding, 'masked_pos') else None)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, Union[Encoding, ]]:
        img, boxes, caption = super().__getitem__(i)
        encoding = self.tokenizer.encode(f"[CLS]{caption}[SEP]")
        if self.mlm:
            encoding = self.random_mask_caption(encoding)
        encoding = self.extend_visual_atten(encoding, boxes)
        return img, boxes, encoding