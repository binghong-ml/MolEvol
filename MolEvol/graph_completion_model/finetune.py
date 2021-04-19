import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from MolEvol.common.fingerprint import largest_connected_subgraph
import numpy as np
import sys
import math
from rdkit import Chem, DataStructs
from tqdm import tqdm
from multiobj_rationale.finetune import remove_order
from multiobj_rationale.properties import get_scoring_function
from multiobj_rationale.fuseprop import *
from rdkit.Chem import AllChem
from MolEvol.common import props_short, props_long

from MolEvol.common.parse_args import args


def get_shaping_scores(scores, shaping_lambda=args.shaping_lambda):
    return [math.exp((score - 1) * shaping_lambda) for score in scores]


class SubgraphDataset_with_scores(Dataset):

    def __init__(self, data, avocab, batch_size, num_decode, shaping_scores=None):
        assert shaping_scores is not None
        data = [(x, proportion) for (smiles, proportion) in zip(data, shaping_scores) for x in
                enum_root(smiles, num_decode)]

        self.batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class MoleculeDataset_with_scores(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        init_smiles, final_smiles, s_scores, g_scores = zip(*self.batches[idx])
        init_batch = [Chem.MolFromSmiles(x) for x in init_smiles]
        mol_batch = [Chem.MolFromSmiles(x) for x in final_smiles]
        init_atoms = [mol.GetSubstructMatch(x) for mol, x in zip(mol_batch, init_batch)]
        mol_batch = [MolGraph(x, atoms) for x, atoms in zip(final_smiles, init_atoms)]
        s_scores = [s_scores[i] for i, x in enumerate(mol_batch) if len(x.root_atoms) > 0]
        g_scores = [g_scores[i] for i, x in enumerate(mol_batch) if len(x.root_atoms) > 0]
        coef = sum(g_scores) / len(g_scores)
        g_scores = [x / coef for x in g_scores]
        mol_batch = [x for x in mol_batch if len(x.root_atoms) > 0]
        if len(mol_batch) < len(self.batches[idx]):
            num = len(self.batches[idx]) - len(mol_batch)
            print("MoleculeDataset: %d graph removed" % (num,))
        return MolGraph.tensorize(mol_batch, self.avocab), torch.tensor(s_scores), torch.tensor(g_scores) if len(
            mol_batch) > 0 else None


def decode_rationales(model, rationale_dataset):
    loader = DataLoader(rationale_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    model.eval()
    cand_mols = []
    norm_scores = []
    with torch.no_grad():
        for tup in tqdm(loader):
            init_smiles, norm_s = zip(*tup)
            try:
                final_smiles = model.decode(init_smiles)
            except StopIteration:
                continue
            mols = [(x, y, z) for x, y, z in zip(init_smiles, final_smiles, norm_s) if y and '.' not in y]
            mols = [(x, y, z) for x, y, z in mols if Chem.MolFromSmiles(y).HasSubstructMatch(Chem.MolFromSmiles(x))]
            cand_mols.extend([(x, y) for x, y, z in mols])
            norm_scores.extend([z for x, y, z in mols])
    return cand_mols, norm_scores


def to_fingerprints(mols):
    mols = [Chem.MolFromSmiles(s) for s in mols]
    mols = [x for x in mols if x is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mols]
    return fps


def filter_novel_cand_mols(sgs_tuples, ref_path):
    novel_tuples = []
    pred_actives = []
    for tup in sgs_tuples:
        ra, mol = tup
        pred_actives.append(mol)

    with open(ref_path) as f:
        next(f)
        true_actives = [line.split(',')[0] for line in f]
    # print('number of active reference', len(true_actives))

    true_fps = to_fingerprints(true_actives)
    pred_fps = to_fingerprints(pred_actives)

    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_fps)
        if max(sims) < 0.4:
            novel_tuples.append(sgs_tuples[i])
    print('Novelty Filter: {} -> {}'.format(len(sgs_tuples), len(novel_tuples)))
    _, smiles_list = zip(*novel_tuples)
    return smiles_list


def scr_filter(cand_mols, s_scores, scoring_function, args, epoch):
    if args.novelty_filter:
        novel_corpus = set(filter_novel_cand_mols(cand_mols, args.ref_path))

    rationales, smiles_list = zip(*cand_mols)
    cand_props = scoring_function(smiles_list)
    cand_mols_update = []
    rationale_dist = {remove_order(r): [0, []] for r in rationales}
    valid_rationale_dict = set()
    with open(args.save_dir + '/valid.' + str(epoch), 'w') as f:
        for (init_smiles, final_smiles), prop, s_score in zip(cand_mols, cand_props, s_scores):
            # for (init_smiles, final_smiles), prop in zip(cand_mols, cand_props):
            rationale = remove_order(init_smiles)
            scr = score_func(prop)
            rationale_dist[rationale][0] += scr / args.finetune_num_decode
            rationale_dist[rationale][1].append(final_smiles)
            valid_rationale_dict.add(rationale)
            print(init_smiles, final_smiles, scr, file=f)
            if scr >= args.scr_thres:
                if (not args.novelty_filter) or (args.novelty_filter and final_smiles in novel_corpus):
                    g_score = score_func(prop)
                    g_score = math.exp((g_score - 1) * args.shaping_lambda)
                    cand_mols_update.append((init_smiles, final_smiles, s_score, g_score))
    rkey_list = rationale_dist.keys()
    del_num = 0
    for ra in rkey_list:
        if ra not in valid_rationale_dict:
            del rationale_dist[ra]
            del_num += 1
    print('Delete Invalid Rationales: {}'.format(del_num))
    print('Scr Filter & Novelty: {} -> {}'.format(len(cand_mols), len(cand_mols_update)))

    return cand_mols_update, rationale_dist


norm_p4 = lambda x: (-x / 9) + 10 / 9
invert_p4 = lambda x: 1.0 / x / 0.9 - 1.0 / 9
score_func = lambda x: (x[0] * x[1] * x[2] * invert_p4(x[3])) ** (1 / 4)
qed_sa_func = lambda x: x[0] >= 0.5 and x[1] >= 0.5 and x[2] >= 0.6 and x[3] <= 4.0
normal_func = lambda x: min(x) >= 0.5


def finetune(rationales, model, args, rationale_corpus, all=True,
             scores=None):
    if all:
        args.compare_func = qed_sa_func
        props = props_long
        print('finetune using compare function qed_sa_func')
    else:
        args.compare_func = normal_func
        props = props_short
        print('finetune using compare function normal_func')

    prop_funcs = [get_scoring_function(prop) for prop in props]
    scoring_function = lambda x: list(zip(*[func(x) for func in prop_funcs]))

    shaping_scores = get_shaping_scores(scores)
    rationale_dataset = SubgraphDataset_with_scores(rationales, args.atom_vocab, args.decode_batch_size,
                                                    args.finetune_num_decode,
                                                    shaping_scores=shaping_scores)

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    for epoch in range(args.load_epoch + 1, args.epoch):
        if args.model_noupd:
            break
        print('epoch', epoch)
        sys.stdout.flush()
        # todo: uncomment to run
        cand_mols, s_scores = decode_rationales(model, rationale_dataset)
        cand_mols, rationale_dist = scr_filter(cand_mols, s_scores, scoring_function, args, epoch)

        # torch.save((cand_mols, rationale_dist), '{}/cr'.format(args.exp_dir))
        # print('(cand_mols, rationale_dist) saved to {}'.format('{}/cr'.format(args.exp_dir)))
        # cand_mols, rationale_dist = torch.load('{}/cr'.format(args.exp_dir))
        model_ckpt = (rationale_dist, model.state_dict())
        torch.save(model_ckpt, os.path.join(args.save_dir, f"model.{epoch}"))

        cand_mols = list(set(cand_mols))
        random.shuffle(cand_mols)

        # Update model

        dataset = MoleculeDataset_with_scores(cand_mols, args.atom_vocab, args.batch_size)
        model.train()

        meters = np.zeros(5)
        for total_step, (batch, s_score_batch, g_score_batch) in enumerate(dataset):
            if batch is None: continue

            g_score_batch = g_score_batch.cuda()
            ce, kl_div, wacc, tacc, sacc = model(*batch)
            kl_loss = (kl_div * g_score_batch).mean()
            loss = (ce * g_score_batch).mean() + kl_loss * args.beta

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # meters = meters + np.array([kl_div.mean().item(), loss.item(), wacc * 100, tacc * 100, sacc * 100])
            meters = meters + np.array([kl_loss.item(), loss.item(), wacc * 100, tacc * 100, sacc * 100])

            if (total_step + 1) % args.print_iter == 0:
                meters /= args.print_iter
                print(
                    "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step + 1, args.beta, meters[0], meters[1], meters[2], meters[3], meters[4],
                        param_norm(model),
                        grad_norm(model)))
                sys.stdout.flush()
                meters *= 0

        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
    if args.epoch == 0:
        epoch = 1
    else:
        epoch += 1

    cand_mols, s_scores = decode_rationales(model, rationale_dataset)
    cand_mols, rationale_dist = scr_filter(cand_mols, s_scores, scoring_function, args, epoch)

    model_ckpt = (rationale_dist, model.state_dict())
    torch.save(model_ckpt, os.path.join(args.save_dir, f"model.{epoch}"))

    new_rationale_cnt = 0
    for ra in rationale_dist.keys():
        if ra not in rationale_corpus:
            new_rationale_cnt += 1
            rationale_corpus[ra] = rationale_dist[ra]

    rationale_corpus_updated = {k: v for k, v in
                                sorted(rationale_corpus.items(), key=lambda item: -item[1][0])[:args.topk]}

    print('Update Rationale Corpus with {} New Rationales'.format(new_rationale_cnt))

    # KEY STEP: UPDATE RATIONALES AFTER FINETUNING
    # rationales = list(rationale_dist.keys())
    # rationales = unique_rationales(rationales)

    return rationale_corpus_updated, model, rationale_dist
