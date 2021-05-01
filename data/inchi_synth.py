import os
import json

import fire
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from inchi_image import InChiSyth, svg_to_image, apply_image_noise
from inchi_bbox import get_bbox


def create_synth_det_dataset(inchi_csv, unique_atom_csv, output_dir, max_sample=20000):
    atom_map = {k: 0.4 / len(InChiSyth.RARE_ATOMS) for k in InChiSyth.RARE_ATOMS}
    atom_map['C0'] = 0.6
    syth = InChiSyth(atom_map)

    inchi_df = pd.read_csv(inchi_csv)
    inchi_df = inchi_df.sample(max_sample)
    
    with open(unique_atom_csv, 'r') as f:
        unique_atom = json.load(f)

    for i, row in tqdm(inchi_df.iterrows()):
        # with logger.catch():
        # print(row.InChI)
        rand_mol = syth.random_swap(row.InChI)
        for j, randmz in enumerate([True, True]):
            boxes_anno, svg = get_bbox(rand_mol, unique_atom, rand_svg=randmz)
            image_id = row.image_id if hasattr(row, 'image_id') else f"{i%10}{i:07}"
            relative_path = os.path.join('/'.join(image_id[:3]), f"{image_id}.png")
            sub_dir = os.path.join(output_dir, os.path.dirname(relative_path))
            os.makedirs(sub_dir, exist_ok=True)
            
            img_path = os.path.join(output_dir, relative_path.replace('.png', f'.{j}.png'))
            json_path = os.path.join(output_dir, relative_path.replace('.png', f'.{j}.json'))
            
            svg_img = svg_to_image(svg).astype(np.uint8)
            svg_img = apply_image_noise(svg_img)
            svg_pil = Image.fromarray(svg_img)
            svg_pil.save(img_path)
            det_anno = {
                'image_id': image_id,
                'boxes': boxes_anno,
                'image_width': svg_pil.size[0],
                'image_height': svg_pil.size[1],
            }
            with open(json_path, mode='w') as f:
                json.dump(det_anno, f)


if __name__ == "__main__":
    fire.Fire({
        "create_synth_det_dataset": create_synth_det_dataset,
    })