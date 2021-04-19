from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from MolEvol.graph_completion_model.finetune import score_func

normal_func = lambda x: (x[0]) >= 0.5 and float(x[1]) >= 0.5 and float(x[2]) > 0.6 and float(x[3]) < 4
topk_func = lambda x: score_func(x) >= 0.5


def filter_actives(sgs_tuples, scr_func=topk_func):
    pred_actives = [mol for ra, mol, x, y, qed, sa in sgs_tuples if scr_func((x, y, qed, sa))]
    return pred_actives


def to_fingerprints(mols):
    mols = [Chem.MolFromSmiles(s) for s in mols]
    mols = [x for x in mols if x is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
    return fps


def eval_success_rate(pred_actives, sgs_tuples):
    success_rate = len(pred_actives) / len(sgs_tuples)
    return success_rate


def eval_novelty(pred_actives, ref_path):
    if len(pred_actives) == 0:
        return 0.

    with open(ref_path) as f:
        next(f)
        true_actives = [line.split(',')[0] for line in f]
    # print('number of active reference', len(true_actives))

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)

    fraction_similar = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) >= 0.4:
            fraction_similar += 1

    novelty = 1 - fraction_similar / len(pred_actives)

    return novelty


def filter_novel_actives_v2(pred_actives, ref_path):
    with open(ref_path) as f:
        next(f)
        true_actives = [line.split(',')[0] for line in f]
    # print('number of active reference', len(true_actives))

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)
    novel_actives = []
    fraction_similar = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) < 0.4:
            novel_actives.append(pred_actives[i])
        else:
            fraction_similar += 1

    novelty = 1 - fraction_similar / len(pred_actives)

    return novelty, novel_actives


def filter_novel_actives(pred_actives, ref_path):
    with open(ref_path) as f:
        next(f)
        true_actives = [line.split(',')[0] for line in f]
    # print('number of active reference', len(true_actives))

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)
    novel_actives = []
    total_novelty = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) < 0.4:
            novel_actives.append(pred_actives[i])
            total_novelty += max(sims)

    # novelty = 1 - fraction_similar / len(pred_actives)

    return total_novelty / len(novel_actives), novel_actives


def eval_diversity(pred_actives):
    if len(pred_actives) == 0:
        return 0.

    pred_fps = to_fingerprints(pred_actives)

    similarity = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)

    n = len(pred_fps)
    n_pairs = n * (n - 1) / 2
    diversity = 1 - similarity / n_pairs
    return diversity


def get_qu(sgs_tuples, ref_path):
    novel_tuples = []
    pred_actives = []
    for tup in sgs_tuples:
        ra, mol, x, y, qed, sa = tup
        if topk_func((x, y, qed, sa)):
            # pred_actives = [mol for ra, mol, x, y, qed, sa in sgs_tuples )]
            pred_actives.append(mol)

    with open(ref_path) as f:
        next(f)
        true_actives = set([get_canonical_smiles(line.split(',')[0]) for line in f])

    print('number of active reference', len(true_actives))
    all_set = set()

    for i in range(len(pred_actives)):
        canon_smiles = get_canonical_smiles(pred_actives[i])
        if canon_smiles not in all_set and canon_smiles not in true_actives:
            all_set.add(canon_smiles)
            novel_tuples.append(sgs_tuples[i])
    print('QU {} -> {}'.format(len(sgs_tuples), len(novel_tuples)))

    return len(novel_tuples) / len(sgs_tuples)


def get_qnu(sgs_tuples, ref_path):
    novel_tuples = []
    pred_actives = []
    for tup in sgs_tuples:
        ra, mol, x, y, qed, sa = tup
        if topk_func((x, y, qed, sa)):
            # pred_actives = [mol for ra, mol, x, y, qed, sa in sgs_tuples )]
            pred_actives.append(mol)

    with open(ref_path) as f:
        next(f)
        true_actives = set([get_canonical_smiles(line.split(',')[0]) for line in f])

    print('number of active reference', len(true_actives))
    all_set = set()

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)

    for i in range(len(pred_actives)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        canon_smiles = get_canonical_smiles(pred_actives[i])
        if canon_smiles not in all_set and max(sims) < 0.4:
            all_set.add(canon_smiles)
            novel_tuples.append(sgs_tuples[i])
    print('QNU {} -> {}'.format(len(sgs_tuples), len(novel_tuples)))

    return len(novel_tuples) / len(sgs_tuples)


def eval_all_v4(sgs_tuples, ref_path):
    unique_success_novel_rate = get_qu(sgs_tuples, ref_path)
    gnu_rate = get_qnu(sgs_tuples, ref_path)
    pred_actives = filter_actives(sgs_tuples, scr_func=topk_func)

    success_rate1 = eval_success_rate(pred_actives, sgs_tuples)
    novelty, pred_actives = filter_novel_actives_v2(pred_actives, ref_path)
    diversity2 = eval_diversity(pred_actives)
    return success_rate1, novelty, diversity2, unique_success_novel_rate, gnu_rate


def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol)
    return canon_smiles
