import numpy as np
import sys
from tqdm import tqdm
import random
from functools import partial
from rdkit import Chem
from multiprocessing import Pool
from multiobj_rationale.properties import gsk3_model, jnk3_model
from multiobj_rationale.mcts import mcts
from MolEvol.common.fingerprint import get_fingerprint_as_array, select_atoms, extract_selected_subgraph, \
    extract_selected_subgraph_for_gcn
from MolEvol.common.parse_args import args
from multiobj_rationale.properties import qed_func, sa_func

# silence sklearn warnings
import warnings

warnings.filterwarnings('ignore')


class dual_gsk3_jnk3_model_4prop():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, combine_method='geo'):
        self.gsk3_model = gsk3_model()
        self.jnk3_model = jnk3_model()
        self.qed_model = qed_func()
        self.sa_model = sa_func()
        self.combine_method = combine_method

    def __call__(self, smiles_list):
        scores_gsk3 = self.gsk3_model(smiles_list)
        scores_jnk3 = self.jnk3_model(smiles_list)
        scores_sa = self.sa_model(smiles_list)
        scores_qed = self.qed_model(smiles_list)
        assert self.combine_method == 'geo'
        scores_final = np.float32(
            (scores_gsk3 * scores_jnk3 * scores_qed * (1.0 / scores_sa / 0.9 - 1.0 / 9)) ** (1 / 4))
        assert np.max(scores_final) <= 1, (scores_gsk3, scores_jnk3, scores_sa, scores_qed, scores_final)
        return scores_final


class dual_gsk3_jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, combine_method='geo'):
        self.gsk3_model = gsk3_model()
        self.jnk3_model = jnk3_model()
        self.combine_method = combine_method

    def __call__(self, smiles_list):
        scores_gsk3 = self.gsk3_model(smiles_list)
        scores_jnk3 = self.jnk3_model(smiles_list)

        if self.combine_method == 'geo':
            return np.float32(np.sqrt(scores_gsk3 * scores_jnk3))
        else:
            return np.float32((scores_jnk3 + scores_gsk3) / 2.0)

    def compute_gradient(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp, info = get_fingerprint_as_array(mol)

        fps = [fp]
        for onbit, subgraphs in info.items():
            step_fp = fp.copy()
            step_fp[0, onbit] = 0
            fps.append(step_fp)

        fps = np.concatenate(fps, axis=0)
        scores_gsk3 = np.float32(self.gsk3_model.clf.predict_proba(fps)[:, 1])
        scores_jnk3 = np.float32(self.jnk3_model.clf.predict_proba(fps)[:, 1])
        combined_scores = np.sqrt(scores_gsk3 * scores_jnk3)

        gradient = combined_scores[1:] - combined_scores[0]

        return info, gradient


DM = dual_gsk3_jnk3_model()
QM = dual_gsk3_jnk3_model_4prop()


def local_for_one_molecule(smiles_idx, num_rounds):
    smiles, idx = smiles_idx
    np.random.seed(idx)
    random.seed(idx)
    mol = Chem.MolFromSmiles(smiles)
    rationale_list = []
    if mol.GetNumAtoms() > 50:
        return smiles, [None]

    assert args.l2x_gcn
    if args.l2x_gcn:
        from MolEvol.explainer.explain import L2X_explain
        onbit_list, importance_list = L2X_explain(smiles)

    tot = sum(importance_list)
    importance_list = [x / tot for x in importance_list]

    min_len, max_len = 3, 5  # int(total_len * 0.3)
    candidate_len_population = [i for i in range(min_len, max_len + 1)]
    candidate_len_list = random.choices(candidate_len_population, k=num_rounds)

    rset = set()
    for num_info in candidate_len_list:
        num_info = min(num_info, len(onbit_list))
        selected_atoms = np.random.choice(onbit_list, num_info, p=importance_list, replace=False)
        selected_atoms = set(int(atom) for atom in selected_atoms)
        # print(selected_atoms)
        subgraph_smiles = extract_selected_subgraph_for_gcn(smiles, selected_atoms)
        minimum_smiles = subgraph_smiles
        if minimum_smiles is None:
            continue
        if minimum_smiles not in rset and mol.GetNumAtoms() > Chem.MolFromSmiles(minimum_smiles).GetNumAtoms():
            rationale_list.append(minimum_smiles)
            rset.add(minimum_smiles)

    return smiles, rationale_list


def do_local_search_disc(molecules, args):
    import time
    st = time.time()
    mol_id_list = [(mol, idx) for idx, mol in enumerate(molecules)]

    work_func = partial(
        local_for_one_molecule,
        num_rounds=args.num_local_rounds
    )

    pool = Pool(args.ncpu)
    results = pool.map(work_func, mol_id_list)
    pool.close()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Time Elapsed:', time.time() - st)
    print('Collect Results...')
    sys.stdout.flush()
    gsls_tuples = []

    all_rationales = set()
    orig_dict = {}
    for orig_smiles, rationales in results:
        sr = set(rationales)
        all_rationales.update(sr)
        for r in sr:
            orig_dict[r] = orig_smiles

    print('Unique Results...')
    print('Time Elapsed:', time.time() - st)
    sys.stdout.flush()
    if args.prop4:
        eval_model = QM
    else:
        eval_model = DM

    if None in all_rationales:
        all_rationales.remove(None)
    len_limit = 500000
    total_len = min(len_limit, len(all_rationales))
    all_rationales_uniq = random.sample(all_rationales, k=total_len)
    scr_list = []
    interval = 50000
    for i in range(0, total_len, interval):
        print('Eval {} to {}...'.format(i, i + interval - 1))
        scr_list += list(eval_model(all_rationales_uniq[i:i + interval]).reshape(-1))
    assert len(scr_list) == total_len

    print('Length of Unique:', len(scr_list))
    print('Time Elapsed:', time.time() - st)
    sys.stdout.flush()
    for r_smiles, scr in (zip(tqdm(all_rationales_uniq), scr_list)):
        mol = Chem.MolFromSmiles(r_smiles)
        if mol.GetNumAtoms() <= args.max_atoms and scr >= args.scr_thres:
            gsls_tuples.append((orig_dict[r_smiles], r_smiles, mol.GetNumAtoms(), scr))
            # print(r_smiles, DM([r_smiles])[0])

    return gsls_tuples


if __name__ == '__main__':
    dual_model = dual_gsk3_jnk3_model()
    # dual_model.gsk3_model.feature_importances_

    smiles = 'C=C(C)CS(=O)c1ccc(Nc2nccc(-c3ccccc3)n2)cc1'
    info, gradient = dual_model.compute_gradient(smiles)

    # lower gradient matters
    for i, (onbit, _) in enumerate(info.items()):
        print(f'onbit={onbit}, gradient={gradient[i]}')
