import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from multiobj_rationale.fuseprop import find_clusters, extract_subgraph
import os


def get_fingerprint(mol):
    info = {}
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=info)
    return features_vec, info


def get_fingerprint_as_array(mol):
    features_vec, info = get_fingerprint(mol)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features.reshape(1, -1), info


def explain_fingerprint_bit(mol, vis_dir=None):
    features_vec, info = get_fingerprint(mol)
    print('on bits:', info)

    for onbit, subgraphs in info.items():
        for center, radius in subgraphs:
            print(f'on bit = {onbit}, center = {center}, radius = {radius}')

            if radius > 0:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius=radius, rootedAtAtom=center)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                atoms = list(amap.keys())
                print(atoms)

                if vis_dir is not None:
                    # mfp2_svg = Draw.DrawMorganEnv(mol, atomId=center, radius=radius, useSVG=True)
                    # svg_f = f'bit{onbit}_center{center}_radius{radius}.svg'
                    # with open(os.path.join(vis_dir, svg_f), 'w') as f:
                    #     f.write(mfp2_svg)

                    png_f = f'bit{onbit}_center{center}_radius{radius}.png'
                    Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=atoms)


def select_atoms(mol, selected_bits, vis_dir=None):
    features_vec, info = get_fingerprint(mol)
    # print('on bits:', info)

    selected_atoms = set()
    for onbit, subgraphs in info.items():
        if onbit in selected_bits:
            for center, radius in subgraphs:
                # print(f'on bit = {onbit}, center = {center}, radius = {radius}')
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius=radius, rootedAtAtom=center)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                atoms = list(amap.keys())
                selected_atoms.update(atoms)
                # print(atoms)

                if vis_dir is not None:
                    png_f = f'bit{onbit}_center{center}_radius{radius}.png'
                    Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=atoms)

    return selected_atoms


def find_minimum_subgraph(smiles, selected_atoms, vis_dir=None):
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    selected_clusters = set()
    cluster_votes = {}
    # First iteration: select a cluster when,
    #   1. An atom uniquely belongs to this cluster,
    #   2. Two atoms belong to this cluster.
    for atom in selected_atoms:
        assert len(atom_cls[atom]) > 0
        if len(atom_cls[atom]) == 1:
            selected_clusters.add(atom_cls[atom][0])
        else:
            for cls in atom_cls[atom]:
                if cls not in cluster_votes:
                    cluster_votes[cls] = 0
                cluster_votes[cls] += 1
                if cluster_votes[cls] >= 2:
                    selected_clusters.add(cls)
    # Second iteration: randomly select a cluster for the remaining atoms.
    for atom in selected_atoms:
        selected = False
        for cls in atom_cls[atom]:
            if cls in selected_clusters:
                selected = True
                break
        if not selected:
            selected_clusters.add(atom_cls[atom][0])

    cluster_neighbor = {}
    for i in range(len(clusters)):
        cluster_neighbor[i] = set()
        for atom in clusters[i]:
            cluster_neighbor[i].update(atom_cls[atom])
        cluster_neighbor[i].remove(i)

    # remove degree-1 unselected clusters iteratively
    leaf_clusters = set()
    while True:
        updated = False
        for i in range(len(clusters)):
            if i in selected_clusters or i in leaf_clusters:
                continue
            if len(cluster_neighbor[i]) > 1:
                removable = True
                neighbor_pairs = [(j, k) for j in cluster_neighbor[i] for k in cluster_neighbor[i] if j < k]
                for j, k in neighbor_pairs:
                    if j not in cluster_neighbor[k] or k not in cluster_neighbor[j]:
                        removable = False
                        break
                if not removable:
                    continue

            leaf_clusters.add(i)
            for j in cluster_neighbor[i]:
                cluster_neighbor[j].remove(i)
            updated = True

        if not updated:
            break

    minimum_atoms = set()
    for i in range(len(clusters)):
        if i not in leaf_clusters:
            minimum_atoms.update(clusters[i])

    minimum_smiles, _ = extract_subgraph(smiles, minimum_atoms)
    # print(f'{smiles} --> {minimum_smiles}')

    if vis_dir is not None:
        png_f = f'atoms_selected{len(selected_atoms)}.png'
        Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=selected_atoms)
        png_f = f'atoms_minimum{len(selected_atoms)}.png'
        Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=minimum_atoms)
        png_f = f'atoms_minimum_extracted{len(selected_atoms)}.png'
        Draw.MolToFile(Chem.MolFromSmiles(minimum_smiles), filename=os.path.join(vis_dir, png_f))

    return minimum_smiles


def extract_selected_subgraph_for_gcn(smiles, selected_atoms, vis_dir=None):
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    selected_clusters = set()

    for atom in selected_atoms:
        for cls in atom_cls[atom]:
            selected_clusters.add(clusters[cls])
    # print(selected_clusters)

    for cls in selected_clusters:
        for atom in cls:
            selected_atoms.add(atom)

    minimum_smiles, _ = extract_subgraph(smiles, selected_atoms)
    # print(selected_atoms)
    # print(f'{smiles} --> {minimum_smiles}')
    if vis_dir is not None:
        png_f = f'atoms_selected{len(selected_atoms)}.png'
        Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=selected_atoms)
        # png_f = f'atoms_minimum_extracted{len(selected_atoms)}.png'
        # Draw.MolToFile(Chem.MolFromSmiles(minimum_smiles), filename=os.path.join(vis_dir, png_f))

    return minimum_smiles


def extract_selected_subgraph(smiles, selected_atoms):
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    selected_clusters = []

    for cls in clusters:
        if len(cls) > 2:
            num_selected = 0
            for atom in cls:
                num_selected += atom in selected_atoms
            if num_selected >= 2:
                # print('select the whole aromatic ring since 2 or more atoms are selected')
                selected_clusters.append(cls)

    for cls in selected_clusters:
        for atom in cls:
            selected_atoms.add(atom)

    minimum_smiles, _ = extract_subgraph(smiles, selected_atoms)
    # print(selected_atoms)
    # print(f'{smiles} --> {minimum_smiles}')

    return minimum_smiles


def sample_subgraphs(smiles, num_samples=10, frac=0.5, vis_dir=None):
    mol = Chem.MolFromSmiles(smiles)
    clusters, atom_cls = find_clusters(mol)
    cluster_sizes = [len(cls) for cls in clusters]
    p = np.array(cluster_sizes).astype('float')
    p /= p.sum()

    selected_atoms_list = []

    for n in range(num_samples):
        selected_clusters = np.random.choice(len(clusters), int(frac * len(clusters)), p=p, replace=False)
        selected_atoms = set()
        for i in selected_clusters:
            for j in clusters[i]:
                selected_atoms.add(j)

        minimum_smiles, _ = extract_subgraph(smiles, selected_atoms)
        selected_atoms_list.append(selected_atoms)
        if vis_dir is not None:
            png_f = f'subgraph_{n}.png'
            Draw.MolToFile(mol, filename=os.path.join(vis_dir, png_f), highlightAtoms=selected_atoms)
            png_f = f'subgraph_{n}_extracted.png'
            print(minimum_smiles)
            Draw.MolToFile(Chem.MolFromSmiles(minimum_smiles), filename=os.path.join(vis_dir, png_f))
    return selected_atoms_list


def largest_connected_subgraph(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles.split('.')]
    mol_lens = [mol.GetNumAtoms() if mol is not None else 0. for mol in mols]
    # print(mol_lens)
    max_idx = mol_lens.index(max(mol_lens))
    return Chem.MolToSmiles(mols[max_idx])


if __name__ == '__main__':
    from MolEvol.common import args, mkdir

    vis_dir = args.visualize_dir
    mkdir(vis_dir)

    # <<<<<<< HEAD
    smiles = 'CC=C(C)Oc1cccc(-c2ccnc(Nc3ccccc3)n2)c1'
    mol = Chem.MolFromSmiles(smiles)
    # explain_fingerprint_bit(mol, vis_dir)
    selected_atoms = select_atoms(mol, selected_bits=[389, 1777])
    # minimum_smiles = find_minimum_subgraph(smiles, selected_atoms, vis_dir)

    # sample_subgraphs(smiles, vis_dir=vis_dir)
    print(selected_atoms)

    # before graph completion: obtain smiles string of (possibly disconnected) subgraph
    subgraph_smiles = extract_selected_subgraph(smiles, selected_atoms)

    # after graph completion: obtain largest connected subgraph
    largest_sg_smiles = largest_connected_subgraph(subgraph_smiles)
    print(largest_sg_smiles)
