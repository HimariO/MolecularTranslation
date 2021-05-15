import itertools
import functools
from collections import defaultdict

import torch
import numpy as np
from torch.nn import functional as F
from . import constant


class EncodedBatchCollator:

    def __call__(self, batch):
        all_boxes = []
        all_tokens = []
        shapes = defaultdict(list)
        for data in batch:
            key, img, boxes, encoding = data
            shapes['img'].append(img.shape)  # (c, h, w)
            shapes['boxes'].append(boxes.shape)  # (n, 4)
            shapes['ids'].append(encoding.ids.shape)  # (m,)
            shapes['masked_pos'].append(encoding.masked_pos.shape)  # (m,)
            shapes['masked_ids'].append(encoding.masked_pos.shape)  # (m,)
            # NOTE: we are not appling dynamic padding on masked_ids(masked toekn label)
            #   because its already been pad to fixed size and wont be feed into model(not affect inferece time)
            all_boxes.append(boxes)
            all_tokens.append(encoding.tokens)
        
        max_shape = {
            k: np.asarray(v).max(axis=0)
            for k, v in shapes.items()}
        padded = defaultdict(list)
        sample_keys = []
        for data in batch:
            key, img, boxes, encoding = data
            sample_keys.append(key)

            img_pad = max_shape['img'] - np.asarray(img.shape)
            pad_size = [[0, d] for d in reversed(img_pad)]
            pad_size = functools.reduce(lambda a, b: a + b, pad_size)
            img = F.pad(img, pad_size, value=0)
            padded['img'].append(img)
            
            token_pad = max_shape['ids'] - np.asarray(encoding.ids.shape)
            pad_size = [[0, d] for d in reversed(token_pad)]
            pad_size = functools.reduce(lambda a, b: a + b, pad_size)  # size should be (2,)
            ids = F.pad(encoding.ids, pad_size, value=constant.PAD_TOKEN_ID)
            type_ids = F.pad(encoding.type_ids, pad_size, value=constant.MASKED_TOKEN_ID_PAD)
            padded['ids'].append(ids)
            padded['type_ids'].append(type_ids)
            
            """
            Pad MLM binary mask
            """
            mlm_pad = max_shape['masked_pos'] - np.asarray(encoding.masked_pos.shape)
            mlm_pad_size = [0, mlm_pad[0]]
            masked_pos = F.pad(encoding.masked_pos, mlm_pad_size, value=0)
            padded['masked_pos'].append(masked_pos)

            mlm_pad = max_shape['masked_ids'] - np.asarray(encoding.masked_ids.shape)
            mlm_pad_size = [0, mlm_pad[0]]
            masked_ids = F.pad(encoding.masked_ids, mlm_pad_size, value=0)
            padded['masked_ids'].append(masked_ids)

            """
            Pad 2D multi-modal attention mask
            """
            tk_to_tk_atten_mask = encoding.token_attention_mask[:, :len(encoding.ids)]
            tk_to_img_atten_mask = encoding.token_attention_mask[:, len(encoding.ids):]
            boxes_pad = max_shape["boxes"] - np.asarray(boxes.shape)  # (2,) value should be [>=0, 0]
            
            tk_atten_pad = [0, token_pad[0], 0, token_pad[0]]
            tk_to_tk_atten_mask = F.pad(tk_to_tk_atten_mask, tk_atten_pad)
            tk_img_atten_pad = [0, boxes_pad[0], 0, token_pad[0]]
            tk_to_img_atten_mask = F.pad(tk_to_img_atten_mask, tk_img_atten_pad)

            img_to_img_atten_mask = encoding.img_attention_mask[:, len(encoding.ids):]
            img_to_tk_atten_mask = encoding.img_attention_mask[:, :len(encoding.ids)]
            img_atten_pad = [0, boxes_pad[0], 0, boxes_pad[0]]
            img_to_img_atten_mask = F.pad(img_to_img_atten_mask, img_atten_pad)
            img_tk_atten_pad = [0, token_pad[0], 0, boxes_pad[0]]
            img_to_tk_atten_mask = F.pad(img_to_tk_atten_mask, img_tk_atten_pad)
            attention_mask = torch.cat([
                torch.cat([tk_to_tk_atten_mask, tk_to_img_atten_mask], dim=1),
                torch.cat([img_to_tk_atten_mask, img_to_img_atten_mask], dim=1),
            ], dim=0)
            padded['attention_mask'].append(attention_mask)
            
        batch_idx_boxes = []
        for i, boxes in enumerate(all_boxes):
            batch_idx_boxes.append(
                F.pad(boxes, (1, 0, 0, 0), value=i)
            )
        batch_idx_boxes = torch.cat(batch_idx_boxes, dim=0)
        # import pdb; pdb.set_trace()

        return (
            sample_keys,
            torch.stack(padded['img']),
            all_boxes,
            torch.stack(padded['ids']),
            torch.stack(padded['type_ids']),
            torch.stack(padded['attention_mask']),
            torch.stack(padded['masked_pos']),
            torch.stack(padded['masked_ids']),
            all_tokens,
        )