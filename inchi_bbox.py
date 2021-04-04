# %%

import os
import glob
import json
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import cv2
import pandas as pd
import numpy as np
import lxml.etree as et
import cssutils
from termcolor import colored
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDrawOptions
from scipy.spatial.ckdtree import cKDTree
from xml.dom import minidom
from pqdm.processes import pqdm
from loguru import logger
from PIL import Image

from inchi_image import svg_to_image, apply_image_noise

# %%

def _get_svg_doc(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    options = MolDrawOptions()
    options.useBWAtomPalette()

    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.SetDrawOptions(options)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    doc = minidom.parseString(svg)
    
    return doc, svg


def _get_rand_svg_doc(mol, render_size=300):
    d = Draw.rdMolDraw2D.MolDraw2DSVG(render_size, render_size)
    dm = Draw.PrepareMolForDrawing(mol)
    
    options = MolDrawOptions()
    options.useBWAtomPalette()
    options.additionalAtomLabelPadding = np.random.uniform(0, 0.3)
    options.bondLineWidth = int(np.random.uniform(1, 3))
    options.multipleBondOffset = np.random.uniform(0.05, 0.2)
    # options.rotate = np.random.uniform(0, 360)
    # options.fixedScale = np.random.uniform(0.05, 0.07)
    options.minFontSize = 12
    options.maxFontSize = options.minFontSize + int(np.round(np.random.uniform(0, 12)))
    
    d.SetFontSize(100)
    d.SetDrawOptions(options)
    d.DrawMolecule(mol)
    d.AddMoleculeMetadata(dm)
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
    # print(et.tostring(svg))
    svg = et.tostring(svg).decode()
    doc = minidom.parseString(svg)
    return doc, svg

    

def _get_unique_atom_inchi_and_rarity(inchi):
    """ HELPER FUNCTION - DONT CALL DIRECTLY
    Get the compound unique atom inchi in the format [AtomType+FormalCharge] and a dictionary
    of the metrics taken into account for rarity measures.
    eg: OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N ---> {'C0', 'N1', 'N0', 'O0', 'S0'}
    :param inchi: Inchi. (string)
    :return: set of atom smiles(strings).
    """

    mol = Chem.MolFromInchi(inchi)
    assert mol, f'INVALID Inchi STRING: {inchi}'

    doc, svg = _get_svg_doc(mol)

    # get atom positions in order to oversample hard cases
    atoms_pos = np.array([[int(round(float(path.getAttribute('drawing-x')), 0)),
                           int(round(float(path.getAttribute('drawing-y')), 0))] for path in
                          doc.getElementsByTagName('rdkit:atom')])

    # calculat the minimum distance between atoms in the molecule
    sampling_weights = {}
    xys = atoms_pos
    kdt = cKDTree(xys)
    dists, neighbours = kdt.query(xys, k=2)
    nearest_dist = dists[:, 1]

    # min distance
    sampling_weights['global_minimum_dist'] = 1 / (np.min(nearest_dist) + 1e-12)
    # number of atoms closer than half of the average distance
    sampling_weights['n_close_atoms'] = np.sum(nearest_dist < np.mean(nearest_dist) * 0.5)
    # average atom degree
    sampling_weights['average_degree'] = np.array([a.GetDegree() for a in mol.GetAtoms()]).mean()
    # number of triple bonds
    sampling_weights['triple_bonds'] = sum([1 for b in mol.GetBonds() if b.GetBondType().name == 'TRIPLE'])
    results = [
        ''.join([a.GetSymbol(), str(a.GetFormalCharge())]) 
        for a in mol.GetAtoms()
    ]
    return results, sampling_weights

def create_unique_ins_labels(data, base_path='.'):
    """
    Create a dictionary with the count of each existent atom-smiles in the
    train dataset and a dataframe with the atom-smiles in each compound.
    eg: Inchi dataframe
    :param data: Pandas data frame with columns ['file_name', 'Inchi']. [Pandas DF]
    :param overwrite: overwrite existing JSON file at base_path + '/data/unique_atoms_smiles.json. [bool]
    :param base_path: base path of the environment. [str]
    :return: A dict of counts[dict] and DataFrame of unique atom-smiles per compound.
    """
    inchi_list = data.InChI.to_list()

    # check if file exists
    output_counts_path = base_path + '/unique_atom_smiles_counts.json'
    output_unique_atoms = base_path + '/unique_atoms_per_molecule.csv'
    output_mol_rarity = base_path + '/mol_rarity_train.csv'

    assert type(inchi_list) == list, 'Input Inchi data type must be a LIST'

    n_jobs = max(mp.cpu_count() - 2, 1)

    # get unique atom-smiles in each compound and count for sampling later.
    result = pqdm(inchi_list,
                  _get_unique_atom_inchi_and_rarity,
                  n_jobs=n_jobs, 
                  desc='Calculating unique atom-smiles and rarity')
    result, sample_weights = list(map(list, zip(*result)))
    counts = Counter(x for xs in result for x in xs)

    # save counts
    with open(output_counts_path, 'w') as fout:
        json.dump(counts, fout)

    # save sample weights
    sample_weights = pd.DataFrame.from_dict(sample_weights)
    sample_weights.insert(0, "image_id", data.image_id)
    sample_weights.to_csv(output_mol_rarity, index=False)

    # save unique atoms in each molecule to oversample less represented classes later
    unique_atoms_per_molecule = pd.DataFrame({'Inchi': inchi_list, 'unique_atoms': [set(r) for r in result]})
    unique_atoms_per_molecule.to_csv(output_unique_atoms, index=False)

    # print(f'{color.BLUE}Counts file saved at:{color.END} {output_counts_path}\n' +
    #       f'{color.BLUE}Unique atoms file saved at:{color.END} {output_unique_atoms}')

    return counts, unique_atoms_per_molecule


def get_bbox(inchi, unique_labels, atom_margin=12, bond_margin=10, rand_svg=False):
    """
    Get list of dics with atom-smiles and bounding box [x, y, width, height].
    :param inchi: STR
    :param unique_labels: dic with labels and idx for training.
    :param atom_margin: margin for bbox of atoms.
    :param bond_margin: margin for bbox of bonds.
    :return:
    """
    # replace unique labels to decide with kind of labels to look for
    labels = defaultdict(int)
    labels['H'] = -1
    for k, v in unique_labels.items():
        labels[k] = v

    mol = Chem.MolFromInchi(inchi)
    if rand_svg:
        doc, svg = _get_rand_svg_doc(mol)
    else:
        doc, svg = _get_svg_doc(mol)

    # with open('tmp_svg.svg', mode='w') as f:
    #     f.write(svg)

    # Get X and Y from drawing and type is generated
    # from mol Object, concatenating symbol + formal charge
    svg_atoms = doc.getElementsByTagName('rdkit:atom')
    mol_atoms_type = [
        ''.join([a.GetSymbol(), str(a.GetFormalCharge())])
        for a in mol.GetAtoms()
    ]
    if len(mol_atoms_type) < len(svg_atoms):
        mol_atoms_type += ['H'] * (len(svg_atoms) - len(mol_atoms_type))
    elif len(mol_atoms_type) > len(svg_atoms):
        raise RuntimeError('Whaaaaat?')
    
    atoms_data = [{
            'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
            'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
            'type': at
        } 
        for path, at in zip(svg_atoms, mol_atoms_type)
    ]
    # if any([a['type'] in ['H1', 'H1'] for a in atoms_data]):
    #     print('A')

    annotations = []
    # anotating bonds
    for path in doc.getElementsByTagName('rdkit:bond'):
        # Set all '\' or '/' as single bonds
        
        ins_type = path.getAttribute('bond-smiles')
        if (ins_type == '\\') or (ins_type == '/'):
            ins_type = '-'

        # make bigger margin for bigger bonds (double and triple)
        _margin = bond_margin
        if (ins_type == '=') or (ins_type == '#'):
            _margin *= 1.5

        # creating bbox coordinates as XYWH.
        begin_atom_idx = int(path.getAttribute('begin-atom-idx')) - 1
        end_atom_idx = int(path.getAttribute('end-atom-idx')) - 1
        x = min(atoms_data[begin_atom_idx]['x'], atoms_data[end_atom_idx]['x']) - _margin // 2  # left-most pos
        y = min(atoms_data[begin_atom_idx]['y'], atoms_data[end_atom_idx]['y']) - _margin // 2  # up-most pos
        width = abs(atoms_data[begin_atom_idx]['x'] - atoms_data[end_atom_idx]['x']) + _margin
        height = abs(atoms_data[begin_atom_idx]['y'] - atoms_data[end_atom_idx]['y']) + _margin

        annotation = {
            'bbox':        [int(x), int(y), int(width), int(height)],
            'category_id': labels[ins_type],
            'type': ins_type,
        }
        annotations.append(annotation)

    # annotating atoms
    for atom in atoms_data:
        _margin = atom_margin

        # better to predict close carbons (2 close instances affected by NMS)
        if atom['type'] == 'C0':
            _margin /= 2

        # Because of the hydrogens normally the + sign falls out of the box
        if atom['type'] == 'N1':
            _margin *= 2

        annotation = {
            'bbox': [
                atom['x'] - _margin,
                atom['y'] - _margin,
                _margin * 2,
                _margin * 2
            ],
            'category_id': labels[atom['type']],
            'type': atom['type'],
        }
        annotations.append(annotation)

    return annotations, svg


def build_label_stat(bms_root):
    bms_train = os.path.join(bms_root, 'train')
    train_df = pd.read_csv(os.path.join(bms_root, 'train_labels.csv'))
    if not os.path.exists(os.path.join(bms_root, 'unique_atom_smiles_counts.json')):
        create_unique_ins_labels(train_df, base_path=bms_root)


def get_bbox_shard(args):
    inchi_df, unique_atom = args
    annotations = []
    for i, row in inchi_df.iterrows():
        # with logger.catch():
        try:
            boxes_anno, _ = get_bbox(row.InChI, unique_atom)
            annotations.append({
                'image_id': row.image_id,
                'boxes': boxes_anno,
            })
        except IndexError:
            print(i, row.InChI, 'index error')
    return annotations

def main(bms_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    unique_atom_csv = os.path.join(bms_root, 'unique_atom_smiles_counts.json')
    train_df = pd.read_csv(os.path.join(bms_root, 'train_labels.csv'))
    with open(unique_atom_csv, 'r') as f:
        unique_atom = json.load(f)
        # rand_inchi = train_df.iloc[5].InChI
        # get_bbox('InChI=1S/C21H30O4/c1-12(22)25-14-6-8-20(2)13(10-14)11-17(23)19-15-4-5-18(24)21(15,3)9-7-16(19)20/h13-16,19H,4-11H2,1-3H3/t13-,14+,15+,16-,19-,20+,21+/m1/s1', unique_atom)
        # get_bbox('InChI=1S/C24H50OSi/c1-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-25-26(5,6)24(2,3)4/h14-15H,7-13,16-23H2,1-6H3/b15-14-/i13D2,14D,15D', unique_atom)
        # get_bbox('InChI=1S/C18H22O4/c19-11-15(7-13-3-1-5-17(21)9-13)16(12-20)8-14-4-2-6-18(22)10-14/h1-6,9-10,15-16,19-22H,7-8,11-12H2/t15-,16-/m0/s1/i3D,4D,5D,6D,9D,10D', unique_atom)
        # get_bbox(rand_inchi, unique_atom)
    # train_df = train_df.iloc[:16000]
    n_worker = 8
    n_split = n_worker * 20
    J = 20
    assert len(train_df) >= n_worker * n_split * J
    args = [(train_df.iloc[i::n_split], unique_atom) for i in range(n_split)]
    jargs = [args[i::J] for i in range(J)]
    
    with mp.Pool(n_worker) as pool:
        # results = pool.map(get_bbox_shard, args)
        for j in range(J):
            results = pqdm(
                jargs[j],
                get_bbox_shard,
                n_jobs=n_worker, 
                desc=f'Calculating boxes - batch {j}')
            for i, result in enumerate(results):
                output_file = os.path.join(output_dir, f"bms_bbox.{j}.{i}.json")
                print(colored(f'Save', color='blue'), f' {output_file}')
                with open(output_file, mode='w') as f:
                    json.dump(result, f)


def generate_img_and_box(args):
    inchi_df, unique_atom, name2img, output_dir = args
    for i, row in inchi_df.iterrows():
        # with logger.catch():
        for j, randmz in enumerate([False, True]):
            boxes_anno, svg = get_bbox(row.InChI, unique_atom, rand_svg=randmz)
            
            relative_path = name2img[row.image_id]
            sub_dir = os.path.join(output_dir, os.path.dirname(relative_path))
            os.makedirs(sub_dir, exist_ok=True)
            
            img_path = os.path.join(output_dir, relative_path.replace('.png', f'.{j}.png'))
            json_path = os.path.join(output_dir, relative_path.replace('.png', f'.{j}.json'))
            
            det_anno = {
                'image_id': row.image_id,
                'boxes': boxes_anno,
            }
            svg_img = svg_to_image(svg).astype(np.uint8)
            svg_img = apply_image_noise(svg_img)
            Image.fromarray(svg_img).save(img_path)
            with open(json_path, mode='w') as f:
                json.dump(det_anno, f)


def main_v2(bms_root, output_dir):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    os.makedirs(output_dir, exist_ok=True)
    
    unique_atom_csv = os.path.join(bms_root, 'unique_atom_smiles_counts.json')
    train_df = pd.read_csv(os.path.join(bms_root, 'train_labels.csv'))
    # train_df = train_df.iloc[:1600]
    with open(unique_atom_csv, 'r') as f:
        unique_atom = json.load(f)
    # get_bbox('InChI=1S/C21H30O4/c1-12(22)25-14-6-8-20(2)13(10-14)11-17(23)19-15-4-5-18(24)21(15,3)9-7-16(19)20/h13-16,19H,4-11H2,1-3H3/t13-,14+,15+,16-,19-,20+,21+/m1/s1', unique_atom, rand_svg=True)
    bms_train = os.path.join(bms_root, 'train')
    # name2img = glob.glob(os.path.join(bms_train, '*', '*', '*', '*.png'))
    # name2img = {
    #     os.path.basename(i).replace('.png', ''): i.replace(bms_train, '')
    #     for i in name2img
    # }
    name2img = {
        row.image_id: os.path.join('/'.join(row.image_id[:3]), f"{row.image_id}.png")
        for _, row in train_df.iterrows()
    }

    n_worker = 12
    n_split = n_worker * 10
    J = 10
    assert len(train_df) >= n_worker * n_split * J, f"{len(train_df)} >= {n_worker * n_split * J}"
    args = [
        (train_df.iloc[i::n_split], unique_atom, name2img, output_dir)
        for i in range(n_split)
    ]
    jargs = [args[i::J] for i in range(J)]

    with mp.Pool(n_worker) as pool:
        for j in range(J):
            pqdm(
                jargs[j],
                generate_img_and_box,
                n_jobs=n_worker, 
                desc=f'Calculating boxes - batch {j}')
            # pool.map(generate_img_and_box, jargs[j])

def bbox_json_breakdown(anno_path, bms_root):
    bms_train = os.path.join(bms_root, 'train')
    name2img = glob.glob(os.path.join(bms_train, '*', '*', '*', '*.png'))
    name2img = {
        os.path.basename(i).replace('.png', ''): i
        for i in name2img
    }
    bbox_json_list = glob.glob(os.path.join(anno_path, "*.json"))
    for box_json_file in bbox_json_list:
        print(colored('[Open]', color='green'), box_json_file)
        with open(box_json_file, mode='r') as f:
            bbox_json = json.load(f)
            n = len(bbox_json)
        for j, anno in enumerate(bbox_json):
            img_id = anno['image_id']
            img_path = name2img[img_id]
            img_dir = os.path.dirname(img_path)
            new_anno_path = os.path.join(img_dir, f"{img_id}.json")
            with open(new_anno_path, mode='w') as f:
                json.dump(anno, f)
                # json.dump(anno, f, indent=2, sort_keys=True)
            print(f"[{j}/{n}] ", new_anno_path)
        #     break
        # break

# %%


if __name__ == '__main__':
    bms_root = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation'
    box_dir = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train-bbox'
    # build_label_stat(bms_root)
    # main(bms_root, box_dir)

    det_dataset_dir = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train-det'
    # bbox_json_breakdown(box_dir, bms_root)
    main_v2(bms_root, det_dataset_dir)
