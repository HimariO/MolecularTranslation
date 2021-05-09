import os
import copy
import glob
import json
import math
import random
import pickle
import hashlib
from collections import defaultdict, namedtuple
from typing import Tuple, Union

import cv2
import torch
import numpy as np
import pandas as pd
from loguru import logger
from tokenizers import Tokenizer, Encoding
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EditableEncoding = namedtuple(
    "EditableEncoding",
    ["ids",
    "src_ids",
    "type_ids",
    "tokens",
    "offsets",
    "attention_mask",])

MaskedEncoding = namedtuple(
    "MaskedEncoding",
    ["ids",
    "src_ids",
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
    "src_ids",
    "type_ids",
    "tokens",
    "offsets",
    "token_attention_mask",
    "img_attention_mask",
    "masked_pos",
    "masked_ids",])


class BoxedBMS(Dataset):

    TMP_DIR = os.path.join(os.environ['HOME'], 'bms_tmp')

    def __init__(self, dataset_dir, anno_csv, max_img_size=720, max_cap_len=None) -> None:
        self.dataset_dir = dataset_dir
        self.anno_csv = anno_csv
        self.id2imgdet, self.id2cap = self.get_samples()
        self.imgids = list(self.id2imgdet.keys())
        self.max_img_size = max_img_size
        self.max_cap_len = max_cap_len
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    
    def get_samples(self):
        logger.info(f"[{self.__class__.__name__}] get_samples")
        pat = os.path.join(self.dataset_dir, '**', '*.png')
        img_list = glob.glob(pat, recursive=True)
        logger.info(f"[{self.__class__.__name__}] find {len(img_list)} imgs")
        # det_json_list = [ip.replace(".png", ".json") for ip in img_list]
        
        os.makedirs(self.TMP_DIR, exist_ok=True)
        md5 = hashlib.md5()
        md5.update(self.dataset_dir.encode("utf-8"))
        dir_hash = str(md5.hexdigest())
        cache_file = os.path.join(self.TMP_DIR, f"{dir_hash}_{len(img_list)}.pickle")
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

            for old_cache in glob.glob(os.path.join(self.TMP_DIR, f'{dir_hash}_*.pickle')):
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
        img = np.transpose(img, [2, 0, 1])
        img_tensor = torch.tensor(img).float()
        return self.normalize(torch.clip(img_tensor, 0, 255) / 255)
        # return torch.clip(img_tensor, 0, 255) / 255

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
        return anno, boxes
    
    def cap_img_size(self, img, bbox):
        h, w = img.shape[-2:]
        if max(h, w) > self.max_img_size:
            scale = self.max_img_size / max(h, w)
            bbox *= scale
            img = transforms.functional.resize(img, [int(h * scale), int(w * scale)])
        return img, bbox
    
    def __len__(self):
        return len(self.id2imgdet)
    
    def get_sample(self, i):
        key = self.imgids[i]
        caption = self.id2cap[key]
        img_path, bbox_json_path = self.id2imgdet[key]
        img = self.load_image(img_path)
        _, boxes = self.load_bbox(bbox_json_path)
        img, boxes = self.cap_img_size(img, boxes)
        return key, img, boxes, caption

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, str]:
        data = None
        while data is None:
            try:
                data = self.get_sample(i)
            except:
                key = self.imgids[i]
                logger.warning(f"Get a zero box sample: {key}")
            i += 1
        return data


class EncodedBBMS(BoxedBMS):
    
    max_masked_tokens = 64

    def __init__(self, dataset_dir: str, anno_csv: str, tokenizer: Tokenizer,
                mask_prob=0.4, mlm=True, det_inchi=False, size_bining=False, **kwargs):
        super().__init__(dataset_dir, anno_csv, **kwargs)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mlm = mlm
        self.det_inchi = det_inchi
        # if self.det_inchi:
        #     """
        #     if we disable mlm it will turn all text input token to [MASK]
        #     which make det_inchi option useless
        #     """
        #     assert self.mlm
    
    def create_token_bins(self, bin_size=25):
        id2len = {k: len(v) for k, v in self.id2cap.items()}
        id2idx = {v: i for i, v in enumerate(self.imgids)}
        bins = defaultdict(list)
        for k in self.imgids:
            l = id2len[k]
            bin_id = math.ceil(l / bin_size)
            bins[bin_id].append(id2idx[k])
        logger.info(f"Created {len(bins)} bins with bin size {bin_size}")
        return bins
            
    def random_mask_caption(self,
                            encoding: Union[Encoding, EditableEncoding],
                            is_mlm=True) -> MaskedEncoding:

        def random_continue_idx(a, b):
            i = a
            groups = []
            while i < b:
                group_size = random.randint(3, 8)
                groups.append([i, min(b, i + group_size)])
                i += group_size + 1
            random.shuffle(groups)
            groups = groups[:max(1, int(len(groups) * self.mask_prob))]
            masked_ids = []
            for group in groups:
                masked_ids += list(range(group[0], min(b, group[1] + 1)))
            return masked_ids

        # tokens = encoding.tokens
        input_ids = copy.deepcopy(encoding.ids)
        sep_ind = input_ids.index(self.tokenizer.token_to_id('[SEP]'))
        seq_a_len = len(input_ids)  # NOTE: accounting [SEP] token in between caption and img feat
        masked_pos = torch.zeros(seq_a_len, dtype=torch.int)
        
        # randomly mask words for prediction, ignore [CLS]
        if is_mlm:
            candidate_masked_idx = random_continue_idx(1, sep_ind + 1)
        else:
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
        num_masked = len(candidate_masked_idx)
        masked_idx = candidate_masked_idx[:num_masked]
        masked_idx = sorted(masked_idx)
        # masked_token = [tokens[i] for i in masked_idx]
        if not self.det_inchi:
            masked_ids = [input_ids[t] for t in masked_idx]  # shape: (num_masked,)
        else:
            masked_ids = [encoding.src_ids[t] for t in masked_idx]  # shape: (num_masked,)
        
        for pos in masked_idx:
            if self.det_inchi:
                if random.random() <= 0.1:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    i = random.randint(0, self.tokenizer.get_vocab_size() - 1)
                    input_ids[pos] = i
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass
            else:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    input_ids[pos] = self.tokenizer.token_to_id('[MASK]')
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    i = random.randint(0, self.tokenizer.get_vocab_size() - 1)
                    input_ids[pos] = i
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

        masked_pos[masked_idx] = 1
        # pad masked tokens to the same length
        # masked_ids = [self.tokenizer.token_to_id(t) for t in masked_token]  # shape: (num_masked,)
        # input_ids = [self.tokenizer.token_to_id(t) for t in tokens]

        if self.max_cap_len is not None:
            # NOTE: going this path when we are inference on data with unknown caption length
            ext_len = max(0, self.max_cap_len - len(input_ids))
            if ext_len > 0:
                input_ids += [self.tokenizer.token_to_id('[MASK]')] * ext_len
                ext_ids = encoding.ids + [self.tokenizer.token_to_id('[PAD]')] * ext_len
                ext_tokens = encoding.tokens + ['[PAD]'] * ext_len
                ext_type_ids = encoding.type_ids + [1] * ext_len
                ext_attention_mask = encoding.attention_mask + [1] * ext_len
                masked_pos = torch.cat([masked_pos, torch.ones(ext_len, dtype=torch.int)])
                masked_ids += [self.tokenizer.token_to_id('[PAD]')] * ext_len

                return MaskedEncoding(
                    ids=input_ids,
                    src_ids=ext_ids,
                    tokens=ext_tokens,
                    type_ids=ext_type_ids,
                    attention_mask=ext_attention_mask,
                    offsets=encoding.offsets,
                    masked_pos=masked_pos,
                    masked_ids=masked_ids,
                )
        return MaskedEncoding(
            ids=input_ids,
            src_ids=encoding.ids,
            tokens=encoding.tokens,
            type_ids=encoding.type_ids,
            attention_mask=encoding.attention_mask,
            offsets=encoding.offsets,
            masked_pos=masked_pos,
            masked_ids=masked_ids,
        )
    
    def extend_visual_atten(
            self,
            encoding: MaskedEncoding,
            boxes) -> VisAttenEncoding:
        atten_1d = encoding.attention_mask
        """
        [CLS][CAPTION][SEP] [BOX_FEAT]
        | atten_1d         | extened part|
        """
        a = len(atten_1d)
        b = len(boxes)
        n = a + b
        attention_mask = torch.zeros([n, n], dtype=torch.long)
        # import pdb; pdb.set_trace()
        attention_mask[:a, :a] = torch.tril(torch.ones((a, a), dtype=torch.long))
        attention_mask[a:, a:] = 1
        attention_mask[:a, a:] = 1
        # attention_mask = attention_mask.float()
        
        return VisAttenEncoding(
            ids=torch.tensor(encoding.ids),
            src_ids=torch.tensor(encoding.src_ids),
            tokens=encoding.tokens,
            type_ids=torch.tensor(encoding.type_ids),
            token_attention_mask=attention_mask[:a],
            img_attention_mask=attention_mask[a:],
            offsets=encoding.offsets,
            masked_ids=torch.tensor(encoding.masked_ids) if hasattr(encoding, 'masked_ids') else None,
            masked_pos=encoding.masked_pos if hasattr(encoding, 'masked_pos') else None)
    
    def get_sample(self, i):
        key = self.imgids[i]
        caption = self.id2cap[key]
        img_path, bbox_json_path = self.id2imgdet[key]
        img = self.load_image(img_path)
        det_anno, boxes = self.load_bbox(bbox_json_path)
        img, boxes = self.cap_img_size(img, boxes)

        if self.det_inchi:
            det_inchi = det_anno['inchi']
            return key, img, boxes, caption, det_inchi
        else:
            return key, img, boxes, caption
    
    def fuse_cap_det_inchi(self, src_encoding: Encoding, det_inchi: str) -> EditableEncoding:
        det_encoding: Encoding = self.tokenizer.encode(f"[CLS]{det_inchi}[SEP]")
        pad_id = self.tokenizer.token_to_id('[PAD]')
        pad_size = len(src_encoding.ids) - len(det_encoding.ids)
        det_ids = det_encoding.ids

        if pad_size > 0:
            pad_det_ids = det_ids + [pad_id] * pad_size
            return EditableEncoding(
                ids=pad_det_ids,
                src_ids=src_encoding.ids,
                tokens=src_encoding.tokens,
                type_ids=src_encoding.type_ids,
                attention_mask=src_encoding.attention_mask,
                offsets=src_encoding.offsets,
            )
        else:
            pad_size = abs(pad_size)

            src_ids = src_encoding.ids + [pad_id] * pad_size
            tokens = src_encoding.tokens + ['[PAD]'] * pad_size
            type_ids = src_encoding.type_ids + [1] * pad_size
            attention_mask = src_encoding.attention_mask + [1] * pad_size

            return EditableEncoding(
                ids=det_ids,
                src_ids=src_ids,
                tokens=tokens,
                type_ids=type_ids,
                attention_mask=attention_mask,
                offsets=src_encoding.offsets,
            )

    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, Union[Encoding, EditableEncoding]]:
        data = super().__getitem__(i)
        key, img, boxes, caption = data[:4]
        encoding = self.tokenizer.encode(f"[CLS]{caption}[SEP]")
        
        if self.det_inchi:
            assert len(data) == 5
            det_inchi = data[4]
            encoding = self.fuse_cap_det_inchi(encoding, det_inchi)
        
        encoding = self.random_mask_caption(encoding, is_mlm=self.mlm)
        encoding = self.extend_visual_atten(encoding, boxes)
        return key, img, boxes, encoding