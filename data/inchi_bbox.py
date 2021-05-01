# %%

import os
import glob
import json
import pickle
import random
import shutil
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
    
    d.SetFontSize(40)
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
    
    assert not os.path.exists(output_counts_path)
    assert not os.path.exists(output_unique_atoms)
    assert not os.path.exists(output_mol_rarity)
    assert type(inchi_list) == list, 'Input Inchi data type must be a LIST'

    n_jobs = max(mp.cpu_count() - 4, 1)

    # get unique atom-smiles in each compound and count for sampling later.

    with mp.Pool(n_jobs) as pool:
        if len(inchi_list) > 1_000_000:
            result = []
            for ai, a in enumerate(range(0, len(inchi_list), 1_000_000)):
                sub_list = inchi_list[a: a + 1_000_000]
                sub_res = pqdm(
                    sub_list,
                    _get_unique_atom_inchi_and_rarity,
                    n_jobs=n_jobs, 
                    desc=f'Calculating unique atom-smiles and rarity - {ai}')
                result += sub_res
        else:
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

    mol = Chem.MolFromInchi(inchi) if isinstance(inchi, str) else inchi
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
            
            svg_img = svg_to_image(svg).astype(np.uint8)
            svg_img = apply_image_noise(svg_img)
            svg_pil = Image.fromarray(svg_img)
            svg_pil.save(img_path)
            det_anno = {
                'image_id': row.image_id,
                'boxes': boxes_anno,
                'image_width': svg_pil.size[0],
                'image_height': svg_pil.size[1],
            }
            with open(json_path, mode='w') as f:
                json.dump(det_anno, f)


def build_bms_det_dataset(bms_root, output_dir):
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


def build_extra_det_dataset(train_label, unique_atom_csv, output_dir):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_label)
    # train_df = train_df.iloc[:1600]
    with open(unique_atom_csv, 'r') as f:
        unique_atom = json.load(f)
    name2img = {
        row.image_id: os.path.join('/'.join(row.image_id[:3]), f"{row.image_id}.png")
        for _, row in train_df.iterrows()
    }

    n_worker = 4
    n_split = n_worker * 4
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


def count_bbox_type(det_dataset):
    """
    {
        '-': 93665516,
        '=': 31876144,
        'C0': 87155530,
        'N0': 12776640,
        'O0': 11997642,
        'F0': 1869434,
        'I0': 58946,
        'S0': 1860626,
        '#': 336700,
        'Cl0': 1075260,
        'Br0': 426102,
        'H': 219538,
        'B0': 13016,
        'Si0': 44308,
        'H0': 13530,
        'P0': 39384,
        'Cl3': 90,
        'O-1': 780,
        'O1': 1244,
        'Si-1': 1312,
        'C-1': 966,
        'C1': 282,
        'B1': 1286,
        'N1': 288,
        'N-1': 400,
        'Si1': 144,
        'F1': 8,
        'P1': 34,
        'B-1': 62,
        'I3': 6,
        'S-1': 10,
        'S1': 6,
        'P-1': 14,
        'Br1': 2,
        'H1': 2,
        'Cl2': 2
    }
    """
    name2json = glob.glob(os.path.join(det_dataset, '*', '*', '*', '*.json'))
    type_cnt = defaultdict(lambda: 0)
    type2sample = defaultdict(set)

    for i, js_path in enumerate(name2json):
        if not js_path.endswith('.0.json'):
            continue
        if i % 100 == 0:
            print(f"{i}/{len(name2json)}")
        with open(js_path, mode='r') as f:
            anno = json.load(f)
            for ins in anno['boxes']:
                type_cnt[ins['type']] += 1
                type2sample[ins['type']].add(anno['image_id'])
    with open('type2sample.pickle', mode='wb') as f:
        pickle.dump(dict(type2sample), f)
    print(type_cnt)


def select_det_by_cls(type2same_pickle, src_dir, dst_dir):
    with open(type2same_pickle, 'rb') as f:
        type2sample = pickle.load(f)
    type_sam_pair = [(k, v) for k, v in type2sample.items()]
    type_sam_pair = sorted(type_sam_pair, key=lambda x: len(x[1]))

    sample_set = set()
    prev_size = 0
    for cls, samples in type_sam_pair:
        if len(samples) < 2048:
            sample_set.update(samples)
        else:
            samples = list(samples)
            random.shuffle(samples)
            sample_set.update(samples[:4096])
        print(f"{cls}:\t{len(sample_set)}\t+{len(sample_set) - prev_size}")
        prev_size = len(sample_set)
    
    for sample in sample_set:
        for ft in ['json', 'png']:
            for i in range(2):
                src_file = os.path.join(src_dir, '/'.join(sample[:3]), f'{sample}.{i}.{ft}')
                dst_file = os.path.join(dst_dir, '/'.join(sample[:3]), f'{sample}.{i}.{ft}')
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)


def filter_label_by_atom(labels_dir):
    cnts = os.path.join(labels_dir, 'unique_atom_smiles_counts.json')
    sample_atom = os.path.join(labels_dir, 'unique_atoms_per_molecule.csv')
    with open(cnts, mode='r') as f:
        cnts = json.load(f)
    sample_atom = pd.read_csv(sample_atom)
    white_list_atoms = [k for k, v in cnts.items() if v <= 1000]
    # print(white_list_atoms)
    # import pdb; pdb.set_trace()
    indices = []
    for i, row in sample_atom.iterrows():
        if i % 1000 == 0:
            print(f"{i}/{len(sample_atom)}")
        check = [a in row.unique_atoms for a in white_list_atoms]
        check = any(check)
        if check:
            indices.append(i)
    rare_samples = sample_atom.iloc[indices]
    rare_samples.to_csv(os.path.join(labels_dir, 'rare_molecule.csv'))


def gather_extra_samples(dir_path, output_file):
    sample_csv_list = glob.glob(os.path.join(dir_path, '*/rare_molecule.csv'))
    sample_csv_list = [pd.read_csv(c) for c in sample_csv_list]
    sample_csv = pd.concat(sample_csv_list, axis=0)
    img_idx = [f"e{i:08}" for i in range(len(sample_csv))]
    sample_csv = pd.DataFrame({
        'image_id': img_idx,
        'InChI': sample_csv.Inchi
    })
    sample_csv.to_csv(output_file)



# %%


if __name__ == '__main__':

    def standar_dataset_preprocess():
        bms_root = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation'
        box_dir = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train-bbox'
        build_label_stat(bms_root)

    def build_standar_dataset_bbox():
        det_dataset_dir = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/det-dataset/train-det'
        det_sampled_dataset_dir = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/det-dataset/train-sample-det'
        # bbox_json_breakdown(box_dir, bms_root)
        # build_bms_det_dataset(bms_root, det_dataset_dir)
        # count_bbox_type(det_dataset_dir)
        select_det_by_cls('/home/ron/Projects/MolecularTranslation/type2sample.pickle', det_dataset_dir, det_sampled_dataset_dir)
    
    def extra_inchi_data():
        extra_csv = "/home/ron/Downloads/bms-molecular-translation/extra/extra_inchis_id.csv"
        extra_sample_pd = pd.read_csv(extra_csv)
        extra_dir = os.path.dirname(extra_csv)
        
        N = 1_000_000
        for ai, a in enumerate(range(0, len(extra_sample_pd), N)):
            sub_dir = os.path.join(extra_dir, str(ai))
            os.makedirs(sub_dir, exist_ok=True)
            create_unique_ins_labels(extra_sample_pd.iloc[a: a+N], base_path=sub_dir)
    
    def sample_extra_inchi():
        sample_sub_dirs = [
            f for f in glob.glob("/home/ron/Downloads/bms-molecular-translation/extra/*")
            if os.path.isdir(f)]
        for dir in sample_sub_dirs:
            print(dir)
            filter_label_by_atom(dir)

    # extra_inchi_data()
    # sample_extra_inchi()
    gather_extra_samples(
        "/home/ron/Downloads/bms-molecular-translation/extra/",
        "/home/ron/Downloads/bms-molecular-translation/extra/rare_extra_inchi.csv")
    build_extra_det_dataset(
        '/home/ron/Downloads/bms-molecular-translation/extra/rare_extra_inchi.csv',
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/unique_atom_smiles_counts.json',
        '/home/ron/Downloads/bms-molecular-translation/extra/det-dataset',
    )