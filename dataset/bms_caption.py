from data.inchi_bbox import bbox_json_breakdown
import os
import glob
import json

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BoxedBMS(Dataset):

    def __init__(self, dataset_dir, anno_csv) -> None:
        self.dataset_dir = dataset_dir
        self.anno_csv = anno_csv
        self.id2imgdet, self.id2cap = self.get_samples()
        self.imgids = list(self.id2cap.keys())
    
    def get_samples(self):
        pat = os.path.join(self.dataset_dir, '**', '*.png')
        img_list = glob.glob(pat, recursive=True)
        # det_json_list = [ip.replace(".png", ".json") for ip in img_list]
        id2imgdet = {
            os.path.basename(img).replace('.png', ''): (
                img, img.replace(".png", ".json"))
            for img in img_list
        }
        anno_pd = pd.read_csv(self.anno_csv)
        id2cap = {row.image_id: row.InChI for _, row in anno_pd.iterrows()}
        assert len(id2imgdet) == len(id2cap)
        return id2imgdet, id2cap
    
    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_bbox(self, json_path):
        with open(json_path, mode='r') as f:
            anno = json.load(f)
        boxes_xywh = torch.tensor([b['bbox'] for b in anno['boxes']])
        boxes_x2y2 = boxes_xywh[:, :2] + boxes_xywh[:, 2:]
        boxes = torch.cat([boxes_xywh[:2], boxes_x2y2], dim=1)
        return boxes
    
    def __len__(self):
        return len(self.id2cap)

    def __getitem__(self, i):
        key = self.imgids[i]
        caption = self.id2cap[key]
        img_path, bbox_json_path = self.id2imgdet[key]
        img = self.load_image(img_path)
        boxes = self.load_bbox(bbox_json_path)
        return img, boxes, caption