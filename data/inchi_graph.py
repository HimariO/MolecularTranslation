import re
import os
import copy
import glob
import json
import multiprocessing as mp
import queue

import torch
import numpy as np
import pandas as pd
import torchvision as tv
from loguru import logger
from rdkit import Chem
from scipy.spatial import cKDTree


def xyhw2xyxy(box):
    return [
        box[0],
        box[1],
        box[0] + box[2],
        box[1] + box[3],
    ]


def bond_nms(bond_boxes_list):
    bbox = torch.tensor(bond_boxes_list['bbox']).float()
    scores = torch.tensor(bond_boxes_list['scores']).float()
    keeps = tv.ops.nms(bbox, scores, iou_threshold=0.5)
    return {
        'bbox': bbox[keeps].int().cpu().numpy(),
        'scores': scores[keeps].cpu().numpy(),
        'type': [bond_boxes_list['type'][i] for i in keeps],
    }


def bbox_to_graph(output, threshold=0.4):
    # calculate atoms mask (pred classes that are atoms/bonds)
    anno = output['boxes']
    anno = [box for box in anno if box['score'] > threshold]
    # get atom list
    atoms_list = [box for box in anno if box['type'] not in ["-", "=", "#", "H"]]  # HACK: filter out implict H atom
    atoms_list = pd.DataFrame({'atom': [a['type'] for a in atoms_list],
                               'x':    [a['bbox'][0] + a['bbox'][2] / 2 for a in atoms_list],
                               'y':    [a['bbox'][1] + a['bbox'][3] / 2 for a in atoms_list]})
    # print(atoms_list)

    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bond_boxes_list = [box for box in anno if box['type'] in ["-", "=", "#"]]
    bond_boxes_list = {
        'bbox': np.asarray([xyhw2xyxy(box['bbox']) for box in bond_boxes_list]),
        'scores': np.asarray([box['score'] for box in bond_boxes_list]),
        'type': [box['type'] for box in bond_boxes_list],
    }
    bond_boxes_list = bond_nms(bond_boxes_list)
    bonds_list = []

    # get bonds
    for bbox, bond_type in zip(bond_boxes_list['bbox'],
                                bond_boxes_list['type'],):

        if bond_type == '-':
            _margin = 5
        else:
            _margin = 8

        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        # print(f"{atoms_list['atom'].iloc[begin_idx]} --{bond_type}-- {atoms_list['atom'].iloc[end_idx]}")
        bonds_list.append((begin_idx, end_idx, bond_type))

    return atoms_list.atom.values.tolist(), bonds_list


def mol_from_graph(atoms, bonds):
    """ construct RDKIT mol object from atoms, bonds and bond types
    atoms: list of atom symbols+fc. ex: ['C0, 'C0', 'O-1', 'N1']
    bonds: list of lists of the born [atom_idx1, atom_idx2, bond_type]
    """

    # create and empty molecular graph to add atoms and bonds
    mol = Chem.RWMol()
    nodes_idx = {}
    bond_types = {'-':   Chem.rdchem.BondType.SINGLE,
                  '=':   Chem.rdchem.BondType.DOUBLE,
                  '#':   Chem.rdchem.BondType.TRIPLE,
                  'AROMATIC': Chem.rdchem.BondType.AROMATIC}
    # bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
    #               'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
    #               'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
    #               'AROMATIC': Chem.rdchem.BondType.AROMATIC}

    # add nodes
    pattern = re.compile(r'([a-zA-Z]{1,3})(\-?\d+)?')
    for idx, node in enumerate(atoms):
        # neutral formal charge
        m = re.match(pattern, node)
        # if ('0' in node) or ('1' in node):
        #     a = node[:-1]
        #     fc = int(node[-1])
        # if '-1' in node:
        #     a = node[:-2]
        #     fc = -1
        # create atom object
        assert m is not None
        a = m.group(1)
        fc = 0 if m.group(2) is None else int(m.group(2))
        a = Chem.Atom(a)
        a.SetFormalCharge(fc)

        # add atom to molecular graph (return the idx in object)
        atom_idx = mol.AddAtom(a)
        nodes_idx[idx] = atom_idx

    # add bonds
    # print(bonds)
    existing_bonds = set()
    prev_mol = copy.deepcopy(mol)
    for idx_1, idx_2, bond_type in bonds:
        if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
            if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                try:
                    mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                except:
                    continue
        existing_bonds.add((idx_1, idx_2))

        if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
        # if Chem.MolFromInchi(Chem.MolToInchi(mol.GetMol())):
            # save safe structure
            prev_mol = copy.deepcopy(mol)
        # check if last addition broke the molecule
        else:
            print(f'invalid: {atoms[idx_1]} -- {bond_type} -- {atoms[idx_2]}')
            if prev_mol is None:
                raise RuntimeError()
            else:
                # load last structure
                mol = copy.deepcopy(prev_mol)

    mol = mol.GetMol()
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return mol


@logger.catch(reraise=True)
def det_to_inchi(annotations, queue=None):
    for anno in annotations:
        # print(anno)
        # fname = os.path.basename(anno)
        with open(anno, mode='r') as f:
            box_anno = json.load(f)
        det_graph = bbox_to_graph(box_anno)
        mol = mol_from_graph(*det_graph)
        inchi_str = Chem.MolToInchi(mol)
        box_anno['inchi'] = inchi_str
        with open(anno, mode='w') as f:
            json.dump(box_anno, f)
        if queue is not None:
            queue.put(anno)


def generate_det_inchi(dataset_dir, workers=16):
    logger.add('generate_det_inchi_{time}.log')
    logger.info(f'dataset_dir: {dataset_dir}')
    annotations = glob.glob(os.path.join(dataset_dir, '**/*.json'), recursive=True)
    split_anno = [(annotations[i::workers],) for i in range(workers)]
    res_queue = mp.Queue()
    worker = [
        mp.Process(
            target=det_to_inchi,
            args=split_anno[i],
            kwargs={'queue': res_queue},
            daemon=True)
        for i in range(workers)]
    for w in worker: w.start()
    
    res_cnt = 0
    while any([w.is_alive() for w in worker]) or not res_queue.empty():
        _ = res_queue.get()
        res_cnt += 1
        if res_cnt % 100 == 0:
            logger.info(f"{res_cnt}/{len(annotations)}")
        if res_cnt >= len(annotations):
            break


def debug():
    import json
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    
    # box_json = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val/f/0/1/f01a81d161bf.json'
    # box_json = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val/e/0/b/e0b70eea865c.json'
    box_json = '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val/e/0/b/e0bbed1ac9c7.json'
    with open(box_json, mode='r') as f:
        box_anno = json.load(f)
    det_graph = bbox_to_graph(box_anno)
    mol = mol_from_graph(*det_graph)
    atoms, bonds = det_graph
    print('atoms: ', atoms)
    print('bonds: ', bonds)

    mol_np = Draw.MolToImage(mol)
    plt.imshow(mol_np)
    plt.show()


if __name__ == '__main__':
    generate_det_inchi('/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train')
    # debug()