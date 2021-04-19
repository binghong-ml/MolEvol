import math
import numpy as np
import rdkit
from rdkit import RDLogger

from MolEvol.common.parse_args import args
from MolEvol.common import update_args, get_properties
from MolEvol.common.eval import eval_all_v4
from MolEvol.graph_completion_model import *
from MolEvol.graph_completion_model.finetune import score_func

import torch


def evaluate_molecules(g_list, s_list=None):
    sg_pairs = []
    if s_list is None:
        for g in g_list:
            sg_pairs.append((None, g))
    else:
        for g, s in zip(g_list, s_list):
            sg_pairs.append((s, g))

    print('Get Properties...')
    if os.path.exists('{}/sgs_tuples_weighted'.format(args.iter_dir)):
        sgs_tuples = torch.load('{}/sgs_tuples_weighted'.format(args.iter_dir))
    else:
        sgs_tuples = get_properties(sg_pairs)
        sgs_tuples = sorted(sgs_tuples, key=lambda x: -score_func((x[2], x[3], x[4], x[5])))
        torch.save(sgs_tuples, '{}/sgs_tuples_weighted'.format(args.iter_dir))
    print('Get Statistics...')

    ans_str = '{} Our Metric {}\n'.format('#' * 10, '#' * 10)
    a, b, c, d, e = eval_all_v4(sgs_tuples, args.ref_path)
    ans_str += 'Success rate: %f\n' % a
    ans_str += 'Novelty: %f\n' % b
    ans_str += 'Diversity: %f\n' % c
    ans_str += 'GU: %f\n' % d
    ans_str += 'GNU: %f\n' % e

    for topk in [args.eval_num // 2]:
        tmp_res = sgs_tuples[:topk]
        _, _, x, y, qed, sa = zip(*tmp_res)
        scr = [score_func((x_i, y_i, qed_i, sa_i)) for x_i, y_i, qed_i, sa_i in zip(x, y, qed, sa)]

        ans_str += '{} Top {} Stats {}\n'.format('=' * 12, topk, '=' * 12)
        ans_str += 'Mean of (x, y, qed, sa, score): {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(
            sum(x) / topk, sum(y) / topk, sum(qed) / topk, sum(sa) / topk, sum(scr) / topk
        )

    print(ans_str)
    return ans_str


def eval_molevol(rationale_all_dist):
    eval_dict = {k: (math.exp(v[0]), v[1]) for k, v in
                 sorted(rationale_all_dist.items(), key=lambda item: -item[1][0])[:300]}
    total_score = sum([v[0] for k, v in eval_dict.items()])
    eval_dict = {k: (round(v[0] / total_score * args.eval_num), v[1]) for k, v in eval_dict.items()}
    print(args.iter_dir)
    g_list = []
    s_list = []
    for k, v in eval_dict.items():
        decode_len = len(v[1])
        s_list += [k] * v[0]
        if decode_len > v[0]:
            g_list += random.sample(v[1], k=v[0])
        else:
            g_list += v[1] + random.choices(v[1], k=v[0] - decode_len)
    ans_str = evaluate_molecules(g_list, s_list)

    output = '{}/result_overall_weighted.txt'.format(args.iter_dir)
    with open(output, 'w') as f:
        f.write(ans_str)


def main(args):
    update_args(args, iter_num=0)
    for la in range(1, args.last_round + 1):
        update_args(args, iter_num=la)
        rationale_all_dist = torch.load('{}/rationale_all_dist'.format(args.iter_dir))
        eval_molevol(rationale_all_dist)


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
