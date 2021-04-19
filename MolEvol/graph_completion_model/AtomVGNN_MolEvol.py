import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from multiobj_rationale.fuseprop.mol_graph import MolGraph
from multiobj_rationale.fuseprop.encoder import GraphEncoder
# from MolEvol.graph_completion_model.decoder_ours import GraphDecoder_ours
from MolEvol.graph_completion_model.decoder_MolEvol import GraphDecoder_ours
from multiobj_rationale.fuseprop.nnutils import *
import numpy as np


def make_cuda(graph_tensors):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors


class AtomVGNN_ours(nn.Module):

    def __init__(self, args):
        super(AtomVGNN_ours, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depth)
        self.decoder = GraphDecoder_ours(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size,
                                         args.latent_size, args.diter)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

    def encode(self, graph_tensors):
        graph_vecs = self.encoder(graph_tensors)
        graph_vecs = [graph_vecs[st: st + le].sum(dim=0) for st, le in graph_tensors[-1]]
        return torch.stack(graph_vecs, dim=0)

    def decode(self, init_smiles):
        batch_size = len(init_smiles)
        z_graph_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode(z_graph_vecs, init_smiles)

    def rsample(self, z_vecs, W_mean, W_var, mean_only=False):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)

        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var), dim=-1)
        if mean_only:
            return z_mean, kl_loss
        else:
            epsilon = torch.randn_like(z_mean).cuda()
            z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
            EPS = 1e-8

            def gaussian_likelihood(x, mu, log_std):
                pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
                return pre_sum.sum(axis=-1)

            z_log_std = z_log_var / 2
            pz_log = gaussian_likelihood(z_vecs, z_mean, z_log_std)
            return z_vecs, kl_loss, pz_log

    def forward(self, graphs, tensors, init_atoms, orders):
        # print(len(orders))
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div, pz_log = self.rsample(graph_vecs, self.G_mean, self.G_var)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)

        # loss is -log p, need to divide batch_size

        return loss, kl_div, wacc, tacc, sacc

    def forward_old(self, graphs, tensors, init_atoms, orders):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div, pz_log = self.rsample(graph_vecs, self.G_mean, self.G_var)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return loss

    def test_reconstruct(self, graphs, tensors, init_atoms, orders, init_smiles):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div = self.rsample(graph_vecs, self.G_mean, self.G_var, mean_only=True)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return self.decoder.decode(z_graph_vecs, init_smiles)

    def likelihood(self, graphs, tensors, init_atoms, orders):
        tensors = make_cuda(tensors)
        graph_vecs = self.encode(tensors)
        z_graph_vecs, kl_div = self.rsample(graph_vecs, self.G_mean, self.G_var, mean_only=True)
        loss, wacc, tacc, sacc = self.decoder(z_graph_vecs, graphs, tensors, init_atoms, orders)
        return -loss - kl_div  # Important: loss is negative log likelihood
