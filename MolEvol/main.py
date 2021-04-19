from shutil import copyfile
import numpy as np
import rdkit
from rdkit import RDLogger
from MolEvol.local_search import do_local_search_disc

from MolEvol.common.parse_args import args
from MolEvol.common import update_args
from MolEvol.graph_completion_model import *
from MolEvol.graph_completion_model.finetune import score_func

from multiobj_rationale.properties import get_scoring_function
from MolEvol.graph_completion_model.finetune import filter_novel_cand_mols, remove_order, qed_sa_func, props_long


def scr_filter(cand_mols):
    args.compare_func = qed_sa_func
    props = props_long
    prop_funcs = [get_scoring_function(prop) for prop in props]
    scoring_function = lambda x: list(zip(*[func(x) for func in prop_funcs]))

    if args.novelty_filter:
        novel_corpus = set(filter_novel_cand_mols(cand_mols, args.ref_path))

    rationales, smiles_list = zip(*cand_mols)
    cand_props = scoring_function(smiles_list)
    cand_mols_update = []
    rationale_dist = {remove_order(r): [0, []] for r in rationales}
    valid_rationale_dict = set()
    with open(args.save_dir + '/eval_decode.0', 'w') as f:
        for (init_smiles, final_smiles), prop in zip(cand_mols, cand_props):
            rationale = remove_order(init_smiles)
            scr = score_func(prop)
            rationale_dist[rationale][0] += scr / args.finetune_num_decode
            rationale_dist[rationale][1].append(final_smiles)
            valid_rationale_dict.add(rationale)
            print(init_smiles, final_smiles, scr, file=f)
            if scr >= args.scr_thres:
                if (not args.novelty_filter) or (args.novelty_filter and final_smiles in novel_corpus):
                    cand_mols_update.append(final_smiles)
    rkey_list = rationale_dist.keys()
    for ra in rkey_list:
        if ra not in valid_rationale_dict:
            del rationale_dist['key']

    return cand_mols_update, rationale_dist


def get_mol_corpus(rationales, model, args):
    all_sg_pairs = complete_rationales_candmol(
        rationales=rationales,
        model=model,
        decode_args=args
    )

    candmol, _ = scr_filter(all_sg_pairs)
    return candmol


def get_topk_rationales(gsls_tuples, topk):
    sorted_data = sorted(gsls_tuples, key=lambda x: -float(x[3]))
    data = sorted_data[:topk]
    _, top_rationales, _, P = zip(*data)
    if args.disc_ok:
        _, all_rationales, _, all_P = zip(*(data[:args.all_topk]))
        cc = 0
        for ra, p in zip(all_rationales, all_P):
            if '.' in ra and cc < 10:
                print(ra, p)
                cc += 1
        print('Total Discrete Rationales: {}'.format(cc))

    print('Number of Top: {}'.format(len(top_rationales)))
    print('Average Score: {}'.format(sum(P) / len(P)))
    sys.stdout.flush()
    return top_rationales, P


def search_rationales(seed_molecules):
    if args.local_search and args.disc_ok:
        print('Explainable Local Search...')
        gsls_tuples = do_local_search_disc(seed_molecules, args)
    top_rationales, P = get_topk_rationales(gsls_tuples, args.topk)
    return top_rationales, P


def collect_mol(mol_file_list):
    mol_corpus = []
    for mol_file in mol_file_list:
        with open(mol_file) as f:
            next(f)
            for line in f.readlines():
                mol_corpus.append(line.split(',')[0])
    return mol_corpus


from MolEvol.eval import eval_molevol


def main(args):
    update_args(args, iter_num=0)
    copyfile(src=args.init_model, dst=args.model_file)
    model = load_model_ours(model_args=args)
    mol_file_list = ['rationaleRL/data/gsk3/actives.txt',
                     'rationaleRL/data/jnk3/actives.txt']
    mol_corpus = collect_mol(mol_file_list)
    rationale_corpus = {}  # each value: (avg_mol_score, [mols that generated])
    rationale_all_dist = {}
    if args.last_round != 0:
        update_args(args, iter_num=args.last_round)
        mol_corpus = torch.load('{}/mol_corpus'.format(args.iter_dir))
        model.load_state_dict(torch.load('{}/model.{}'.format(args.iter_dir, args.epoch))[1])
        rationale_corpus = torch.load('{}/rationale_corpus'.format(args.iter_dir))
        rationale_all_dist = torch.load('{}/rationale_all_dist'.format(args.iter_dir))
        print('Loading last round checkpoints...')
    for i in range(args.last_round + 1, args.rounds + 1):
        print('round %d' % i)
        update_args(args, iter_num=i)

        rationales, P = search_rationales(mol_corpus)

        rationale_corpus, model, rationale_dist = finetune(rationales, model, args, rationale_corpus, scores=P)
        rationale_all_dist.update(rationale_dist)

        eval_molevol(rationale_all_dist)

        mol_corpus = get_mol_corpus(rationale_corpus.keys(), model, args)
        torch.save(rationale_dist, '{}/rationale_dist'.format(args.iter_dir))
        print('rationale_dist saved to {}'.format('{}/rationale_dist'.format(args.save_dir)))
        torch.save(rationale_all_dist, '{}/rationale_all_dist'.format(args.iter_dir))
        print('rationale_all_dist saved to {}'.format('{}/rationale_all_dist'.format(args.save_dir)))
        torch.save(mol_corpus, '{}/mol_corpus'.format(args.iter_dir))
        print('mol_corpus saved to {}'.format('{}/mol_corpus'.format(args.save_dir)))
        torch.save(rationale_corpus, '{}/rationale_corpus'.format(args.iter_dir))
        print('rationale_corpus saved to {}'.format('{}/rationale_corpus'.format(args.save_dir)))


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.ncpu = 20
    args.novelty_filter = True
    args.disc_ok = True
    args.prop4 = True
    args.local_search = True
    args.l2x_gcn = True
    args.exp_dir = 'experiment/[MolEvol]{}*{}'.format(args.rounds, args.epoch)

    assert args.disc_ok
    args.init_model = 'models/chembl-molgen/model.25'
    print(args)

    os.makedirs(args.exp_dir, exist_ok=True)
    cmd = 'cp -r MolEvol {}'.format(args.exp_dir)
    print("Moving main files to {}...".format(args.exp_dir))
    os.system(cmd)
    with open('{}/config'.format(args.exp_dir), 'w') as g:
        g.write(str(args))

    main(args)
