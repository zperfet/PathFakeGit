# -*- coding: utf-8 -*-

__doc__ = """Tree GRU aka Recursive Neural Networks."""

import torch.nn.functional as F
import torch.nn as nn
from config import *
from VAE.pretrained_vae import *
from get_args import _ma_vae_args


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None


################################# generate tree structure ##############################
def gen_nn_inputs(root_node, ini_word):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    root_node.idx = 0
    X_word, X_index = [root_node.word], [root_node.index]
    tree, internal_word, internal_index, leaf_idxs = _get_tree_path(root_node)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'),
            np.array(leaf_idxs, dtype='int32')
            )


def _get_tree_path(root_node):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], [], [0]
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer
    tree = []
    word = []
    index = []
    leafs = []
    idx_cnt = root_node.idx
    for layer in layers:
        for node in layer:
            if not node.children:
                leafs.append(node.idx)
                continue
            for child in node.children:
                idx_cnt += 1
                child.idx = idx_cnt
                tree.append([node.idx, child.idx])
                word.append(child.word if child.word is not None else -1)
                index.append(child.index if child.index is not None else -1)
    return tree, word, index, leafs


############################### tree rnn class ######################################
class RvNN(nn.Module):
    def __init__(self, word_dim, hidden_dim=5, Nclass=4,
                 degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True):
        super(RvNN, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        self.degree = degree  # 这里比较奇怪的是，在创建模型的时候是没有对degree进行赋值的
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.device = _ma_vae_args.cuda

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.word_dim]))
        self.W_z_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_z_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_z_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]))
        self.W_r_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_r_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_r_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]))
        self.W_h_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_h_td = nn.parameter.Parameter(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h_td = nn.parameter.Parameter(self.init_vector([self.hidden_dim]))
        self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.hidden_dim]))
        self.b_out_td = nn.parameter.Parameter(self.init_vector([self.Nclass]))

        self._vae = PretrainedVAE(model_archive=vampire_pre_model_path,
                                  device=self.device,
                                  background_frequency=vampire_bgfreq_path,
                                  requires_grad=False,
                                  scalar_mix=None,
                                  dropout=0.5)
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim)
        self.vae_res = dict()

    def get_vae_output(self, input):
        if input not in self.vae_res.keys():
            self.vae_res[input] = self._vae(input)
        return self.vae_res[input]

    def forward(self, tree, edge, leaf_idxs, y):
        if tree.dim() == 1:
            tree = torch.unsqueeze(tree, 0)
        final_state = self.compute_tree_states(tree, edge, leaf_idxs)
        pred, loss = self.predAndLoss(final_state, y)
        return pred, loss

    def recursive_unit(self, child_text, parent_h):
        child_xe = self.E_td[:, child_text].sum(dim=1)
        child_xe = torch.unsqueeze(child_xe, 0)
        # child_xe2 = self._vae(torch.unsqueeze(child_text, 0))
        # child_xe = torch.cat([child_xe, child_xe2], -1)
        # child_xe = self.layer_norm(child_xe)
        z_td = torch.sigmoid(self.W_z_td.mul(child_xe).sum(dim=1) + self.U_z_td.mul(parent_h).sum(dim=1) + self.b_z_td)
        r_td = torch.sigmoid(self.W_r_td.mul(child_xe).sum(dim=1) + self.U_r_td.mul(parent_h).sum(dim=1) + self.b_r_td)
        c = torch.tanh(self.W_h_td.mul(child_xe).sum(dim=1) + self.U_h_td.mul(parent_h * r_td).sum(dim=1) + self.b_h_td)
        h_td = z_td * parent_h + (1 - z_td) * c
        return h_td

    def compute_tree_states(self, tree, edge, leaf_idxs):
        def _recurrence(tree_node, edge_pair, node_h):
            parent_h = node_h[edge_pair[0]]
            child_h = self.recursive_unit(tree_node, parent_h)
            node_h = torch.cat((node_h, child_h.view(1, -1)))
            return node_h

        # node_h = torch.zeros([1, self.hidden_dim], device=device)
        # 计算root的隐含状态
        node_h = self.E_td[:, tree[0]].sum(dim=1)
        node_h = torch.unsqueeze(node_h, 0)
        # node_h2 = self._vae(torch.unsqueeze(tree[0], 0))
        # node_h = torch.cat([node_h, node_h2], -1)
        # node_h = self.layer_norm(node_h)
        for tree_node_info, edge_pair in zip(tree[1:], edge[1:]):
            node_h = _recurrence(tree_node_info, edge_pair, node_h)
        return node_h[leaf_idxs].max(dim=0)[0]
        # return node_h

    def predAndLoss(self, final_state, ylabel):
        pred = F.softmax(self.W_out_td.mul(final_state).sum(dim=1) + self.b_out_td, dim=-1)
        # loss = (torch.tensor(ylabel, dtype=torch.float) - pred).pow(2).sum()
        loss = (ylabel - pred).pow(2).sum()
        return pred, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, tree, edge, leaf_idxs):
        if tree.dim() == 1:
            tree = torch.unsqueeze(tree, 0)
        final_state = self.compute_tree_states(tree, edge, leaf_idxs)
        return F.softmax(self.W_out_td.mul(final_state).sum(dim=1) + self.b_out_td, dim=-1)
