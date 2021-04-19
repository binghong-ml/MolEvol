import os, pickle

VOCAB_DICT = {'B': 0, 'Br': 1, 'C': 2, 'Cl': 3, 'F': 4, 'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
NUM_NODE_LABEL = len(VOCAB_DICT)
invert_p4 = lambda x: 1.0 / x / 0.9 - 1.0 / 9
score_func = lambda x: (x[0] * x[1] * x[2] * invert_p4(x[3])) ** (1 / 4)

import rdkit.Chem as Chem
import numpy as np

MAX_ATOMS = 50


def get_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    if atom_num > MAX_ATOMS:
        return None

    adj = np.array(Chem.GetAdjacencyMatrix(mol))
    adj_np = np.zeros((MAX_ATOMS, MAX_ATOMS))
    adj_np[:atom_num, :atom_num] = adj
    feat_np = []
    for i, atom in enumerate(mol.GetAtoms()):
        # print(atom.GetSymbol())
        node_label = VOCAB_DICT[atom.GetSymbol()]
        feat_np.append(node_label)
        # node_label_one_hot = [0] * NUM_NODE_LABEL
        # node_label_one_hot[node_label] = 1
        # feat_np[i] = np.array(node_label_one_hot)
    feat_np = np.array(feat_np)
    return feat_np, adj_np
