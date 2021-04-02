import torch.nn.functional as F
import torch.nn as nn
from VAE.pretrained_vae import *
from get_args import _ag_args
from config import *
from allennlp.training.metrics import CategoricalAccuracy


class AGMLP(nn.Module):
    def __init__(self, word_dim, random_dim=100,
                 vae_dim=81, class_num=4,
                 degree=2, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=True,
                 ):
        super(AGMLP, self).__init__()
        self.word_dim = word_dim
        self.random_dim = random_dim
        self.vae_dim = vae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.device = _ag_args.cuda

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.word_dim, self.random_dim]))
        self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
        # self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.random_dim + self.vae_dim]))
        self.b_out_td = nn.parameter.Parameter(self.init_vector([self.class_num]))

        self._vae = PretrainedVAE(model_archive=vampire_pre_model_path,
                                  device=self.device,
                                  background_frequency=vampire_bgfreq_path,
                                  requires_grad=False,
                                  scalar_mix=None,
                                  dropout=0.5)
        self.dropout = torch.nn.Dropout(0.3)
        self.w_random = nn.parameter.Parameter(self.init_matrix([self.random_dim, self.random_dim]))
        self.b_random = nn.parameter.Parameter(self.init_matrix([self.random_dim]))
        self.w_vae = nn.parameter.Parameter(self.init_matrix([self.vae_dim, 50]))
        self.b_vae = nn.parameter.Parameter(self.init_matrix([50]))
        if _ag_args.novae:
            self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
        else:
            self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
        self.b_out = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(self, x, y):
        if x.dim() == 1: x = torch.unsqueeze(x, 0)
        final_state = self.compute_tree_states(x)
        prob, loss = self.predAndLoss(final_state, y)
        return prob, loss

    def compute_tree_states(self, x):
        # 计算root的隐含状态
        path_random = self.E_td[x, :]
        # path_random = torch.mm(path_random, self.w_random) + self.b_random
        if _ag_args.novae:
            path_hidden = path_random
        else:
            path_vae = self._vae(x)
            path_vae = (path_vae.unsqueeze(0)
                        .expand(x.shape[1], x.shape[0], -1)
                        .permute(1, 0, 2).contiguous())
            path_hidden = path_vae
            # path_vae = torch.mm(path_vae, self.w_vae) + self.b_vae
            # path_hidden = torch.cat([path_random, path_vae], -1)
        # (P,D181)*(D181,C4)=>(P,C4)
        path_hidden = self.dropout(path_hidden)
        path_hidden = path_hidden.sum(1)
        path_hidden = path_hidden / (x.shape[0] * 0.6)
        # path_hidden = path_hidden.max(dim=0)[0]
        # path_hidden = self.dropout(path_hidden)
        path_pred = self.w_out.mul(path_hidden).sum(dim=1) + self.b_out
        # path_prob = F.softmax(path_hidden, dim=-1)
        return path_pred

    def predAndLoss(self, pred, ylabel):
        # loss = (ylabel - pred).pow(2).sum()
        loss = self._loss(pred.unsqueeze(0), ylabel.long())
        prob = F.softmax(pred, -1)
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x):
        if x.dim() == 1: x = torch.unsqueeze(x, 0)
        final_state = self.compute_tree_states(x)
        return final_state
