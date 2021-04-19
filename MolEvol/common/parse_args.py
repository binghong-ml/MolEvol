import argparse
import os
from multiobj_rationale.fuseprop import common_atom_vocab

parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu', type=int, default=3)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1)

# ============ graph completion model ============== #
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model_file', default=None)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=400)
parser.add_argument('--embed_size', type=int, default=400)
parser.add_argument('--decode_batch_size', type=int, default=20)
parser.add_argument('--latent_size', type=int, default=20)
parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--diter', type=int, default=3)

# ================== rationaleRL =================== #
parser.add_argument('--init_model', type=str, default=os.path.join('models/chembl-molgen/model.25'))

# ==================== finetune ==================== #
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--clip_norm', type=float, default=20.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--finetune_num_decode', type=int, default=200)
parser.add_argument('--novelty_filter', action='store_true')
parser.add_argument('--model_noupd', action='store_true')

parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--anneal_rate', type=float, default=1.0)
parser.add_argument('--print_iter', type=int, default=50)

parser.add_argument('--extra', action='store_true')
parser.add_argument('--last_round', type=int, default=0)

# ==================== evaluation =================== #
parser.add_argument('--ref_path', type=str, default=os.path.join(os.path.dirname(__file__),
                                                                 '../../rationaleRL/data/dual_gsk3_jnk3/actives.txt'))

# =================== local search ================== #
parser.add_argument('--topk', type=int, default=200)
parser.add_argument('--all_topk', type=int, default=2000)
# parser.add_argument('--seed_molecule_path', default=None)
parser.add_argument('--combine_method', default='geo', choices=['geo', 'arith'])
parser.add_argument('--rollout', type=int, default=20)
parser.add_argument('--c_puct', type=float, default=10)
parser.add_argument('--max_atoms', type=int, default=20)
parser.add_argument('--min_atoms', type=int, default=15)
parser.add_argument('--prop_delta', type=float, default=0.5)
parser.add_argument('--ncand', type=int, default=5)
parser.add_argument('--ncpu', type=int, default=20)
parser.add_argument('--prop4', action='store_true')

parser.add_argument('--eval_iter', type=int, default=None)
parser.add_argument('--num_local_rounds', type=int, default=500)
parser.add_argument('--local_search', action='store_true')
parser.add_argument('--l2x_gcn', action='store_true')

parser.add_argument('--shaping_lambda', type=float, default=5)

parser.add_argument('--disc_ok', action='store_true', help='If it is okay for disconnect fingerprint')
# =================== score filter ================== #
parser.add_argument('--scr_thres', type=float, default=0.5)

# =================== evaluation ================ #
parser.add_argument('--eval_novel', action='store_true')
parser.add_argument('--eval_num', type=int, default=20000)

# =================== visualizations ================ #
parser.add_argument('--visualize_dir', default=os.path.join(os.path.dirname(__file__), '../../visualizations'))

args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
