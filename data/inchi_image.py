import os
import re
import copy
import random
import itertools
import functools
from pathlib import Path
from io import BytesIO
from typing import Dict, List

import numpy as np
import pandas as pd
import lxml.etree as et
import cssutils
import cairosvg
from PIL import Image
from skimage.transform import resize

# import IPython
# from IPython.display import SVG
# from IPython.display import display
# import ipywidgets as widgets

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawOptions
from torch._C import Value


def one_in(n):
    return np.random.randint(n) == 0 and True or False


def yesno():
    return one_in(2)


def svg_to_image(svg, convert_to_greyscale=True):
    if isinstance(svg, str):
        svg_str = svg
    else:
        svg_str = et.tostring(svg)
    # TODO: would prefer to convert SVG dirrectly to a numpy array.
    png = cairosvg.svg2png(bytestring=svg_str)
    image = np.array(Image.open(BytesIO(png)), dtype=np.float32)
    # Naive greyscale conversion.
    if convert_to_greyscale:
        image = image.mean(axis=-1)
    return image


def apply_image_noise(mol_np_img):
    assert mol_np_img.ndim == 2
    mask = np.random.uniform(size=[*mol_np_img.shape[:2]])
    # mask = np.abs(np.random.normal(size=[*mol_np_img.shape[:2], 1]))
    mask = (mask > .1).astype(np.float32)
    noise =  np.random.uniform(size=[*mol_np_img.shape[:2]]) < .9995
    noise = noise.astype(np.float32)

    masked_mol = (255 - mol_np_img.astype(np.float32)) * mask
    masked_mol = ((255 - masked_mol) * noise).astype(np.uint8)
    return masked_mol


def elemstr(elem):
    return ', '.join([item[0] + ': ' + item[1] for item in elem.items()])


# Streches the value range of an image to be exactly 0, 1, unless the image appears to be blank.
def stretch_image(img, blank_threshold=1e-2):
    img_min = img.min()
    img = img - img_min
    img_max = img.max()
    if img_max < blank_threshold:
        # seems to be blank or close to it
        return img
    img_max = img.max()
    if img_max < 1.0:
        img = img/img_max
    return img


def random_molecule_image(inchi, drop_bonds=True, add_noise=True, render_size=1200, margin_fraction=0.2):
    # Note that the original image is returned as two layers: one for atoms and one for bonds.
    #mol = Chem.MolFromSmiles(smiles)
    mol = Chem.inchi.MolFromInchi(inchi)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(render_size, render_size)
    
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.additionalAtomLabelPadding = np.random.uniform(0, 0.3)
    options.bondLineWidth = int(np.random.uniform(1, 4))
    options.multipleBondOffset = np.random.uniform(0.05, 0.2)
    options.rotate = np.random.uniform(0, 360)
    options.fixedScale = np.random.uniform(0.05, 0.07)
    options.minFontSize = 20
    options.maxFontSize = options.minFontSize + int(np.round(np.random.uniform(0, 36)))
    
    d.SetFontSize(100)
    d.SetDrawOptions(options)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg_str = d.GetDrawingText()
    # Do some SVG manipulation
    svg = et.fromstring(svg_str.encode('iso-8859-1'))
    
    atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    bond_elems = svg.xpath(r'//svg:path[starts-with(@class,"bond-")]', namespaces={'svg': 'http://www.w3.org/2000/svg'})
    # Change the font.
    font_family = np.random.choice([
        'serif',
        'sans-serif'
    ])
    
    for elem in atom_elems:
        style = elem.attrib['style']
        css = cssutils.parseStyle(style)
        css.setProperty('font-family', font_family)
        css_str = css.cssText.replace('\n', ' ')
        elem.attrib['style'] = css_str
    # Create the original image layers.
    # TODO: separate atom and bond layers
    bond_svg = copy.deepcopy(svg)
    # remove atoms from bond_svg
    
    for elem in bond_svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_bond_img = svg_to_image(bond_svg)
    
    atom_svg = copy.deepcopy(svg)
    # remove bonds from atom_svg
    for elem in atom_svg.xpath(r'//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        parent_elem = elem.getparent()
        if parent_elem is not None:
            parent_elem.remove(elem)
    orig_atom_img = svg_to_image(atom_svg)
    
    if drop_bonds:
        num_bond_elems = len(bond_elems)
        if one_in(3):
            while True:
                # drop a bond
                # Let's leave at least one bond!
                if num_bond_elems > 1:
                    bond_elem_idx = np.random.randint(num_bond_elems)
                    bond_elem = bond_elems[bond_elem_idx]
                    bond_parent_elem = bond_elem.getparent()
                    if bond_parent_elem is not None:
                        bond_parent_elem.remove(bond_elem)
                        num_bond_elems -= 1
                else:
                    break
                if not one_in(4):
                    break
    
    img = svg_to_image(svg) > 254
    img = 1*img  # bool → int
    # Calculate the margins.
    black_indices = np.where(img == 0)
    row_indices, col_indices = black_indices
    if len(row_indices) >= 2:
        min_y, max_y = row_indices.min(), row_indices.max() + 1
    else:
        min_y, max_y = 0, render_size
    if len(col_indices) >= 2:
        min_x, max_x = col_indices.min(), col_indices.max() + 1
    else:
        min_x, max_x = 0, render_size
    margin_size = int(np.random.uniform(0.8*margin_fraction, 1.2*margin_fraction)*max(max_y - min_y, max_x - min_x))
    min_y, max_y = max(min_y - margin_size, 0), min(max_y + margin_size, render_size)
    min_x, max_x = max(min_x - margin_size, 0), min(max_x + margin_size, render_size)
    
    img = img[min_y:max_y, min_x:max_x]
    img = img.reshape([img.shape[0], img.shape[1]]).astype(np.float32)
    
    orig_bond_img = orig_bond_img[min_y:max_y, min_x:max_x]
    orig_atom_img = orig_atom_img[min_y:max_y, min_x:max_x]
    
    scale = np.random.uniform(0.2, 0.4)
    sz = (np.array(orig_bond_img.shape[:2], dtype=np.float32)*scale).astype(np.int32)
    
    orig_bond_img = resize(orig_bond_img, sz, anti_aliasing=True)
    orig_atom_img = resize(orig_atom_img, sz, anti_aliasing=True)
    
    img = resize(img, sz, anti_aliasing=False)
    img = img > 0.5
    if add_noise:
        # Add "salt and pepper" noise.
        salt_amount = np.random.uniform(0, 0.3)
        salt = np.random.uniform(0, 1, img.shape) < salt_amount
        img = np.logical_or(img, salt)
        pepper_amount = np.random.uniform(0, 0.001)
        pepper = np.random.uniform(0, 1, img.shape) < pepper_amount
        img = np.logical_or(1 - img, pepper)
    
    img = img.astype(np.uint8)  # boolean -> uint8
    orig_bond_img = 1 - orig_bond_img/255
    orig_atom_img = 1 - orig_atom_img/255
    # Stretch the range of the atom and bond images so tha tthe min is 0 and the max. is 1
    orig_bond_img = stretch_image(orig_bond_img)
    orig_atom_img = stretch_image(orig_atom_img)
    return img, orig_bond_img, orig_atom_img


# def image_widget(a, greyscale=True):
#     img_bytes = BytesIO()
#     img_pil = Image.fromarray(a)
#     if greyscale:
#         img_pil = img_pil.convert("L")
#     else:
#         img_pil = img_pil.convert("RGB")
#     img_pil.save(img_bytes, format='PNG')
#     return widgets.Image(value=img_bytes.getvalue())


# def test_random_molecule_image(n=4, graphics=True):
#     for imol in range(n):
#         #smiles = np.random.choice(some_smiles)
#         mol_index = np.random.randint(len(TRAIN_LABELS))
#         mol_id, inchi = TRAIN_LABELS['image_id'][mol_index], TRAIN_LABELS['InChI'][mol_index]
#         mol_train_img_path = TRAIN_DATA_PATH / mol_id[0] /mol_id[1] / mol_id[2] / (mol_id + '.png')
#         train_img = Image.open(mol_train_img_path)
        
#         img, orig_bond_img, orig_atom_img = random_molecule_image(inchi)
        
#         if graphics:
#             print('+-------------------------------------------------------------------------------')
#             print(f'Molecule #{imol + 1}: {mol_id}: {inchi}')
#             print('Training image path:', mol_train_img_path)
#             print('Size:', img.shape)
#             combined_orig_img =  np.clip(np.stack([orig_atom_img, orig_bond_img, np.zeros_like(orig_bond_img)], axis=-1), 0.0, 1.0)
#             combined_orig_img = (255*combined_orig_img).astype(np.uint8)
#             widget1 = image_widget(combined_orig_img, greyscale=False)
#             widget2 = image_widget((255*(1 - img)).astype(np.uint8))
#             sidebyside = widgets.HBox([widget1, widget2])
            
#             display(sidebyside)
#             print(f'Image from training data:')
#             print('Size:', train_img.size)
#             display(train_img)
#     return


def create_dataset(bms_root, output_dir):
    # bms_root = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation'
    bms_train = os.path.join(bms_root, 'train')
    anno = os.path.join(bms_root, 'train_labels.csv')
    ann_csv = pd.read_csv(anno)
    for i, row in ann_csv.iterrows():
        inchi_str = row.InChI
        image_id = row.image_id
        img, orig_bond_img, orig_atom_img = random_molecule_image(inchi_str)
        output_path = os.path.join(output_dir, image_id)
        Image.fromarray(img).save(output_path)
        

class InChiSyth:

    ATOM_CHARGE = {
        "N": -3,
        "O": -2,
        "S": -2,
        "F": -1,
        "Cl": -1,
        "I": -1,
        "Br": -1,
        "P": -3,
        "Si": -4,
        "H": -1,
        "B": -3,
        "C": -4,
    }

    RARE_ATOMS = [
        'Cl3',
        'O-1',
        'O1',
        'Si-1',
        'C-1',
        'C1',
        'B1',
        'N1',
        'N-1',
        'Si1',
        'F1',
        'P1',
        'B-1',
        'I3',
        'S-1',
        'S1',
        'P-1',
        'Br1',
        'H1',
        'Cl2',
        'Br2',
        'Cl1',
        'I1',
        'I2',
    ]

    def __init__(self, atom_maps: Dict[str, float], num_atom_range=[10, 30], sparsity_range=[0.1, 0.3]):
        """
        atom_maps: element to sample weight
        sparsity_range: range of how dense atoms is connected together with random bonds
        """
        atoms = list(atom_maps.keys())
        self.num_atom_range = num_atom_range
        self.sparsity_range = sparsity_range

        self.atom_metas = {a: m for a, m in zip(atoms, self.parse_atom_name(atoms))}
        self.atom_maps = atom_maps
        self.bond_maps = {
            Chem.rdchem.BondType.SINGLE: 0.8,
            Chem.rdchem.BondType.DOUBLE: 0.1,
            Chem.rdchem.BondType.TRIPLE: 0.1,
        }

    def parse_atom_name(self, atoms: List[str]):
        pattern = re.compile(r'([a-zA-Z]{1,3})(\-?\d+)?')
        res = []
        for atom in atoms:
            m = re.match(pattern, atom)
            assert m is not None
            ele = m.group(1)
            charge = None if m.group(2) is None else int(m.group(2))
            res.append({
                "type": ele, 
                "charge": charge
            })
        return res
    
    def random_one(self):
        synth_mol = Chem.RWMol()
        N = random.randint(*self.num_atom_range)
        sampled = np.random.choice(
                    list(self.atom_maps.keys()),
                    size=[N],
                    p=list(self.atom_maps.values()))
        for atom in sampled:
            meta = self.atom_metas[atom]
            a = Chem.Atom(meta['type'])
            a.SetFormalCharge(meta['charge'])
            aid = synth_mol.AddAtom(a)

        edge_thred = np.random.uniform(size=[], low=self.sparsity_range[0], high=self.sparsity_range[1])
        adj_mtx = np.random.uniform(size=[N, N], low=0, high=1) < edge_thred
        adj_mtx = np.triu(adj_mtx, k=1)
        adj_mask = 1 - np.triu(np.ones_like(adj_mtx), k=N//4)
        print(adj_mask)
        
        adj_mtx = adj_mtx * adj_mask
        print(adj_mtx.astype(np.int32))

        bond_list = list(self.bond_maps.keys())
        bond_idx = list(range(len(bond_list)))
        for i, j in itertools.product(range(N), range(N)):
            if adj_mtx[i, j]:
                bond = np.random.choice(
                    bond_idx,
                    size=[1],
                    p=list(self.bond_maps.values())
                )[0]
                bond = bond_list[bond]
                print(bond)
                synth_mol.AddBond(i, j, bond)
        del_idx = []
        for i, adj_sum in adj_mtx.sum(-1):
            if adj_sum < 1:
                del_idx.append(i)
        for i in del_idx[::-1]:
            synth_mol.RemoveAtom(i)
        return synth_mol
    
    def random_swap(self, inchi: str):
        src_mol = Chem.MolFromInchi(inchi)
        synth_mol = Chem.RWMol()
        atoms_residual_charge = []
        for ai, atom in enumerate(src_mol.GetAtoms()):
            s = atom.GetSymbol()
            c = atom.GetFormalCharge()
            
            nc = 0
            contain_aromatic = False
            for bond in atom.GetBonds():
                btype = bond.GetBondType()
                if btype == Chem.rdchem.BondType.SINGLE:
                    nc += 1
                elif btype == Chem.rdchem.BondType.DOUBLE:
                    nc += 2
                elif btype == Chem.rdchem.BondType.AROMATIC:
                    nc += 2
                    contain_aromatic = True
                elif btype == Chem.rdchem.BondType.TRIPLE:
                    nc += 3
                else:
                    raise ValueError(f"Unknown bond type: {btype}")
            
            if f"{s}{c}" != 'C0' and not contain_aromatic and nc <= 3:
                meta = random.choice(list(self.atom_metas.values()))
                a = Chem.Atom(meta['type'])
                a.SetFormalCharge(meta['charge'])
                synth_mol.AddAtom(a)

                nc -= meta['charge']
                nc += self.ATOM_CHARGE[meta['type']]
                atoms_residual_charge.append(nc)
                # print(f"{s}{c} -> {meta['type']}{meta['charge']}, nc:{nc}")
            else:
                synth_mol.AddAtom(atom)
                atoms_residual_charge.append(0)
        
        for i, rc in enumerate(atoms_residual_charge):
            if rc != 0:
                rc = abs(rc)
                while rc > 0:
                    a = Chem.Atom("C")
                    a.SetFormalCharge(0)
                    aid = synth_mol.AddAtom(a)
                    synth_mol.AddBond(
                        i,
                        aid,
                        Chem.rdchem.BondType.SINGLE,)
                    rc -= 1
        
        bond_list = list(self.bond_maps.keys())
        bond_idx = list(range(len(bond_list)))
        for bond in src_mol.GetBonds():
            rand_bond = np.random.choice(
                    bond_idx,
                    size=[1],
                    p=list(self.bond_maps.values())
                )[0]
            # rand_bond = bond_list[bond]
            synth_mol.AddBond(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondType(),)
        return synth_mol
