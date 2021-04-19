import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiobj_rationale.fuseprop import *
from MolEvol.graph_completion_model.AtomVGNN_MolEvol import AtomVGNN_ours


def load_model(model_args):
    """Load graph completion model (and rationales) from disk.

    :param model_args: namespace containing atom_vocab, rnn_type, embed_size, hidden_size, depth, latent_size, diter
    :return: rationale list, model
    """
    model = AtomVGNN(model_args).cuda()
    model_ckpt = torch.load(model_args.model_file)
    print('loading pre-trained model from {}'.format(model_args.model_file))
    if type(model_ckpt) is tuple:
        model.load_state_dict(model_ckpt[1])
    else:
        model.load_state_dict(model_ckpt)

    return model


def load_model_ours(model_args):
    """Load graph completion model (and rationales) from disk.

    :param model_args: namespace containing atom_vocab, rnn_type, embed_size, hidden_size, depth, latent_size, diter
    :return: rationale list, model
    """
    model = AtomVGNN_ours(model_args).cuda()
    model_ckpt = torch.load(model_args.model_file)
    print('loading pre-trained model from {}'.format(model_args.model_file))
    if type(model_ckpt) is tuple:
        model.load_state_dict(model_ckpt[1])
    else:
        model.load_state_dict(model_ckpt)

    return model


def load_model_old(model_args):
    """Load graph completion model (and rationales) from disk.

    :param model_args: namespace containing atom_vocab, rnn_type, embed_size, hidden_size, depth, latent_size, diter
    :return: rationale list, model
    """
    model = AtomVGNN(model_args).cuda()
    model_ckpt = torch.load(model_args.model_file)
    assert type(model_ckpt) is tuple
    print('loading model with rationale distribution', file=sys.stderr)
    print('loading model from %s' % model_args.model_file)
    rationales = list(model_ckpt[0].keys())
    model.load_state_dict(model_ckpt[1])
    # else:
    #     print('loading pre-trained model', file=sys.stderr)
    #     testdata = [line.split()[1] for line in open(args.rationale)]
    #     testdata = unique_rationales(testdata)
    #     model.load_state_dict(model_ckpt)

    return rationales, model


# def complete_rationales(rationales, model, decode_args):
#     """Complete molecules from rationales.
#
#     :param rationales: list of rationales
#     :param model: graph completion model
#     :param dataset_args: atom_vocab, batch_size, num_decode
#     :return: sg_pairs: pairs of (rationale, generated molecule)
#     """
#     model.eval()
#
#     print('Completing rationales...')
#     dataset = SubgraphDataset(rationales, decode_args.atom_vocab, decode_args.decode_batch_size, decode_args.num_decode)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
#
#     sg_pairs = []
#     with torch.no_grad():
#         for init_smiles in tqdm(loader):
#             final_smiles = model.decode(init_smiles)
#             for x, y in zip(init_smiles, final_smiles):
#                 if y is not None:
#                     sg_pairs.append((x, y))
# 
#     return sg_pairs

#
# def complete_rationales_2w(rationales, model, decode_args):
#     """Complete molecules from rationales.
#
#     :param rationales: list of rationales
#     :param model: graph completion model
#     :param dataset_args: atom_vocab, batch_size, num_decode
#     :return: sg_pairs: pairs of (rationale, generated molecule)
#     """
#     model.eval()
#
#     ## todo: change this back
#     import math
#     decode_args.num_decode = math.ceil(20000 / len(rationales))
#     dataset = SubgraphDataset(rationales, decode_args.atom_vocab, decode_args.decode_batch_size, decode_args.num_decode)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
#
#     sg_pairs = []
#     with torch.no_grad():
#         for init_smiles in tqdm(loader):
#             final_smiles = model.decode(init_smiles)
#             for x, y in zip(init_smiles, final_smiles):
#                 if y is not None:
#                     sg_pairs.append((x, y))
#
#     return sg_pairs

# def complete_rationales_target(rationales, model, decode_args, target=30000):
#     """Complete molecules from rationales.
#
#     :param rationales: list of rationales
#     :param model: graph completion model
#     :param dataset_args: atom_vocab, batch_size, num_decode
#     :return: sg_pairs: pairs of (rationale, generated molecule)
#     """
#     model.eval()
#
#     ## todo: change this back
#     import math
#     decode_args.num_decode = math.ceil(target / len(rationales))
#     dataset = SubgraphDataset(rationales, decode_args.atom_vocab, decode_args.decode_batch_size, decode_args.num_decode)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
#
#     sg_pairs = []
#     with torch.no_grad():
#         for init_smiles in tqdm(loader):
#             final_smiles = model.decode(init_smiles)
#             for x, y in zip(init_smiles, final_smiles):
#                 if y is not None:
#                     sg_pairs.append((x, y))
#
#     return sg_pairs


def complete_rationales_candmol(rationales, model, decode_args):
    """Complete molecules from rationales.

    :param rationales: list of rationales
    :param model: graph completion model
    :param dataset_args: atom_vocab, batch_size, num_decode
    :return: sg_pairs: pairs of (rationale, generated molecule)
    """
    model.eval()

    # in order to get the same number of candmol like in old method, we need to decode for args.finetune_num_decode times and do filtering
    dataset = SubgraphDataset(rationales, decode_args.atom_vocab, decode_args.decode_batch_size, decode_args.finetune_num_decode)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    sg_pairs = []
    with torch.no_grad():
        for init_smiles in tqdm(loader):
            final_smiles = model.decode(init_smiles)
            for x, y in zip(init_smiles, final_smiles):
                if y is not None:
                    sg_pairs.append((x, y))

    return sg_pairs