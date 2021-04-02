import torch.nn.functional as F
import torch.nn as nn
from VAE.pretrained_vae import *
from get_args import _args
from config import *
from models.WAE import WAEEncoder
from pytorch_transformers import BertModel, BertConfig
import gc


class ResponseWAECat(nn.Module):
    def __init__(self, random_vocab_dim, response_vocab_dim,
                 wae_weight_path, random_dim=50, wae_dim=50,
                 class_num=4, momentum=0.9):
        super(ResponseWAECat, self).__init__()
        self.random_vocab_dim = random_vocab_dim
        self.response_vocab_dim = response_vocab_dim
        self.random_dim = random_dim
        self.wae_dim = wae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.device = _args.cuda
        self.wae_weight_path = wae_weight_path
        self.freeze_wae = _args.freeze_wae
        self.wae_dropout = _args.wae_dropout
        self.wae_encoder = WAEEncoder(self.wae_weight_path, self.response_vocab_dim,
                                      self.device, self.freeze_wae, self.wae_dropout)

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.random_vocab_dim, self.random_dim]))
        self.w_random = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
        self.b_random = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.w_wae = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
        self.b_wae = nn.parameter.Parameter(self.init_matrix([self.class_num]))

        # self.random_layer = nn.Linear(self.random_dim, self.class_num)
        # self.wae_layer = nn.Linear(self.wae_dim, self.class_num)

        self.w_cat = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim + self.wae_dim]))
        self.b_cat = nn.Parameter(self.init_matrix([self.class_num]))

        self.dropout = torch.nn.Dropout(0.3)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(self.wae_dim + self.random_dim)

    def forward(self, x_random, x_response, y):
        pred = self.predict_up(x_random, x_response)
        prob, loss = self.compute_prob_and_loss(pred, y)
        return prob, loss

    def random_predict(self, x):
        path_random = self.E_td[x, :].sum(dim=1)
        path_random = F.leaky_relu(path_random)
        path_random = self.dropout(path_random)
        path_random = path_random.max(dim=0)[0]
        # path_random = self.random_layer(path_random)
        # path_random = F.leaky_relu(path_random)
        path_random = self.w_random.mul(path_random).sum(1) + self.b_random
        path_random = F.softmax(path_random, dim=-1)
        return path_random

    def wae_predict(self, x):
        path_wae = self.wae_encoder(x)
        path_wae = F.leaky_relu(path_wae)
        path_wae = path_wae.max(dim=0)[0]
        # path_wae = self.wae_layer(path_wae)
        # path_wae = F.leaky_relu(path_wae)
        path_wae = self.w_wae.mul(path_wae).sum(1) + self.b_wae
        path_wae = F.softmax(path_wae, dim=-1)
        return path_wae

    def random_wae_cat(self, random_x, wae_x):
        path_random = self.E_td[random_x, :].mean(dim=1)
        path_random = path_random.max(dim=0)[0]
        # path_random = F.leaky_relu(path_random)
        # path_random = self.dropout(path_random)
        path_wae = self.wae_encoder(wae_x)
        path_wae = path_wae.max(dim=0)[0]
        # path_wae = F.leaky_relu(path_wae)
        path_hidden = torch.cat([path_random, path_wae], -1)
        # path_hidden = F.leaky_relu(path_hidden)
        path_hidden = self.dropout1(path_hidden)
        # path_hidden = path_hidden.max(dim=0)[0]
        path_hidden = self.w_cat.mul(path_hidden).sum(1) + self.b_cat
        path_hidden = F.softmax(path_hidden, dim=-1)
        return path_hidden

    # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
    def compute_prob_and_loss(self, pred, label):
        label_num = label.cpu().numpy().tolist().index(1)
        l = torch.Tensor([label_num]).cuda(device).long()
        loss = self._loss(pred.unsqueeze(0), l)
        # prob = F.softmax(pred, -1)
        prob = pred
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_random, x_response):
        if x_random.dim() == 1: x_random = torch.unsqueeze(x_random, 0)
        if x_response.dim() == 1: x_response = torch.unsqueeze(x_response, 0)
        # final_state = self.compute_tree_states(x)
        if _args.novae:
            pred = self.random_predict(x_random)
        elif _args.norandom:
            pred = self.wae_predict(x_response)
        else:
            # random_pred = self.random_predict(x_random)
            # wae_pred = self.wae_predict(x_response)
            # pred = random_pred + wae_pred
            pred = self.random_wae_cat(x_random, x_response)
        return pred


class ResponseWAE(nn.Module):
    def __init__(self, random_vocab_dim, response_vocab_dim,
                 wae_weight_path, random_dim=50, wae_dim=50,
                 class_num=4, momentum=0.9):
        super(ResponseWAE, self).__init__()
        self.random_vocab_dim = random_vocab_dim
        self.response_vocab_dim = response_vocab_dim
        self.random_dim = random_dim
        self.wae_dim = wae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.device = _args.cuda
        self.wae_weight_path = wae_weight_path
        self.freeze_wae = _args.freeze_wae
        self.wae_dropout = _args.wae_dropout
        self.wae_encoder = WAEEncoder(self.wae_weight_path, self.response_vocab_dim,
                                      self.device, self.freeze_wae, self.wae_dropout)

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.random_vocab_dim, self.random_dim]))
        self.w_random = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
        self.b_random = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.w_wae = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
        self.b_wae = nn.parameter.Parameter(self.init_matrix([self.class_num]))

        self.random_layer = nn.Linear(self.random_dim, self.class_num)
        self.wae_layer = nn.Linear(self.wae_dim, self.class_num)

        self.w_cat = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim + self.wae_dim]))
        self.b_cat = nn.Parameter(self.init_matrix([self.class_num]))

        self.dropout = torch.nn.Dropout(0.3)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(self.wae_dim + self.random_dim)

    def forward(self, x_random, x_response, y):
        pred = self.predict_up(x_random, x_response)
        prob, loss = self.compute_prob_and_loss(pred, y)
        return prob, loss

    def random_predict(self, x):
        path_random = self.E_td[x, :].sum(dim=1)
        path_random = F.leaky_relu(path_random)
        path_random = self.dropout(path_random)
        path_random = path_random.max(dim=0)[0]
        # path_random = self.random_layer(path_random)
        # path_random = F.leaky_relu(path_random)
        path_random = self.w_random.mul(path_random).sum(1) + self.b_random
        path_random = F.softmax(path_random, dim=-1)
        return path_random

    def wae_predict(self, x):
        path_wae = self.wae_encoder(x)
        path_wae = F.leaky_relu(path_wae)
        path_wae = path_wae.max(dim=0)[0]
        # path_wae = self.wae_layer(path_wae)
        # path_wae = F.leaky_relu(path_wae)
        path_wae = self.w_wae.mul(path_wae).sum(1) + self.b_wae
        path_wae = F.softmax(path_wae, dim=-1)
        return path_wae

    # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
    def compute_prob_and_loss(self, pred, label):
        label_num = label.cpu().numpy().tolist().index(1)
        l = torch.Tensor([label_num]).cuda(device).long()
        loss = self._loss(pred.unsqueeze(0), l)
        # prob = F.softmax(pred, -1)
        prob = pred
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_random, x_response):
        if x_random.dim() == 1: x_random = torch.unsqueeze(x_random, 0)
        if x_response.dim() == 1: x_response = torch.unsqueeze(x_response, 0)
        # final_state = self.compute_tree_states(x)
        if _args.novae:
            pred = self.random_predict(x_random)
        elif _args.norandom:
            pred = self.wae_predict(x_response)
        else:
            random_pred = self.random_predict(x_random)
            wae_pred = self.wae_predict(x_response)
            pred = random_pred * _args.rate_lambda + wae_pred * (1 - _args.rate_lambda)
        return pred


class ResponseWAEBERT(nn.Module):
    def __init__(self, random_vocab_dim, response_vocab_dim,
                 wae_weight_path, bert_dim=50, wae_dim=50,
                 class_num=4, momentum=0.9):
        super(ResponseWAEBERT, self).__init__()
        self.random_vocab_dim = random_vocab_dim
        self.response_vocab_dim = response_vocab_dim
        self.bert_dim = bert_dim
        self.wae_dim = wae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.device = _args.cuda
        self.wae_weight_path = wae_weight_path
        self.freeze_wae = _args.freeze_wae
        self.wae_dropout = _args.wae_dropout
        self.wae_encoder = WAEEncoder(self.wae_weight_path, self.response_vocab_dim,
                                      self.device, self.freeze_wae, self.wae_dropout)
        config = BertConfig.from_pretrained(bert_weight_type, num_labels=self.class_num)
        self.bert_model = BertModel.from_pretrained(bert_weight_type,
                                                    cache_dir=bert_weight_dir_path,
                                                    config=config)
        self.w_random = nn.parameter.Parameter(self.init_matrix([self.class_num, self.bert_dim]))
        self.b_random = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.w_wae = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
        self.b_wae = nn.parameter.Parameter(self.init_matrix([self.class_num]))

        self.random_layer = nn.Linear(self.bert_dim, self.class_num)
        self.wae_layer = nn.Linear(self.wae_dim, self.class_num)

        self.w_cat = nn.parameter.Parameter(self.init_matrix([self.class_num, self.class_num * 2]))
        self.b_cat = nn.Parameter(self.init_matrix([self.class_num]))

        self.dropout = torch.nn.Dropout(0.3)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(self.wae_dim + self.bert_dim)

    def forward(self, x_random, x_response, y):
        pred = self.predict_up(x_random, x_response)
        prob, loss = self.compute_prob_and_loss(pred, y)
        return prob, loss

    def bert_predict(self, x):
        # path_random = self.E_td[x, :].sum(dim=1)
        # path_random = F.leaky_relu(path_random)
        # path_bert_lst = []
        # for i in range(x.shape[0]):
        #     path_bert_lst.append(self.bert_model(torch.unsqueeze(x[i], 0))[1])
        # path_bert = torch.cat(path_bert_lst, 0)
        # path_bert_lst.clear()
        # gc.collect()
        # torch.cuda.empty_cache()
        path_bert = self.bert_model(x[:10])[1]
        # path_bert = self.dropout(path_bert)
        path_bert = path_bert.max(dim=0)[0]
        # path_random = self.random_layer(path_random)
        # path_random = F.leaky_relu(path_random)
        path_bert = self.w_random.mul(path_bert).sum(1) + self.b_random
        path_bert = F.softmax(path_bert, dim=-1)
        return path_bert

    def wae_predict(self, x):
        path_wae = self.wae_encoder(x)
        path_wae = F.leaky_relu(path_wae)
        path_wae = path_wae.max(dim=0)[0]
        # path_wae = self.wae_layer(path_wae)
        # path_wae = F.leaky_relu(path_wae)
        path_wae = self.w_wae.mul(path_wae).sum(1) + self.b_wae
        path_wae = F.softmax(path_wae, dim=-1)
        return path_wae

    # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
    def compute_prob_and_loss(self, pred, label):
        label_num = label.cpu().numpy().tolist().index(1)
        l = torch.Tensor([label_num]).cuda(device).long()
        loss = self._loss(pred.unsqueeze(0), l)
        # prob = F.softmax(pred, -1)
        prob = pred
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_bert, x_response):
        if x_bert.dim() == 1: x_bert = torch.unsqueeze(x_bert, 0)
        if x_response.dim() == 1: x_response = torch.unsqueeze(x_response, 0)
        # final_state = self.compute_tree_states(x)
        if _args.novae:
            pred = self.bert_predict(x_bert)
        elif _args.norandom:
            pred = self.wae_predict(x_response)
        else:
            random_pred = self.bert_predict(x_bert)
            wae_pred = self.wae_predict(x_response)
            pred = random_pred + wae_pred
        return pred


# random使用全部路径文本；wae使用response路径文本
# 二者使用不同的id编码，即每条路径包含2个id编码
# random和wae的隐变量concat再做分类
# random使用全部路径文本；wae使用response路径文本
# 二者使用不同的id编码，即每条路径包含2个id编码
# random和wae隐变量concat再分类
class ResponseCatWAE(nn.Module):
    def __init__(self, random_vocab_dim, response_vocab_dim,
                 wae_weight_path, random_dim=50, wae_dim=50,
                 class_num=4, momentum=0.9):
        super(ResponseCatWAE, self).__init__()
        self.random_vocab_dim = random_vocab_dim
        self.response_vocab_dim = response_vocab_dim
        self.random_dim = random_dim
        self.wae_dim = wae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.device = _args.cuda
        self.wae_weight_path = wae_weight_path
        self.freeze_wae = _args.freeze_wae
        self.wae_dropout = _args.wae_dropout
        self.wae_encoder = WAEEncoder(self.wae_weight_path, self.response_vocab_dim,
                                      self.device, self.freeze_wae, self.wae_dropout)

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.random_vocab_dim, self.random_dim]))
        self.w_random = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
        self.b_random = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.w_wae = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
        self.b_wae = nn.parameter.Parameter(self.init_matrix([self.class_num]))

        self.random_layer = nn.Linear(self.random_dim, self.class_num)
        self.wae_layer = nn.Linear(self.wae_dim, self.class_num)

        self.w_cat = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim + self.random_dim]))
        self.b_cat = nn.Parameter(self.init_matrix([self.class_num]))

        self.dropout = torch.nn.Dropout(0.3)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(self.wae_dim + self.random_dim)

    def forward(self, x_random, x_response, y):
        if x_random.dim() == 1: x = torch.unsqueeze(x_random, 0)
        if x_response.dim() == 1: x = torch.unsqueeze(x_response, 0)
        if _args.novae:
            # 0.7027
            pred = self.random_hidden(x_random, if_pred=True)
        elif _args.norandom:
            # 0.6588
            pred = self.wae_hidden(x_response, if_pred=True)
        else:
            random_hidden = self.random_hidden(x_random, if_pred=False)
            wae_hidden = self.wae_hidden(x_response, if_pred=False)
            pred = torch.cat([random_hidden, wae_hidden], -1)
            # pred = self.dropout1(pred)
            pred = self.w_cat.mul(pred).sum(1) + self.b_cat
            # pred = self.dropout1(pred)
            # 相加+softmax：0.6588
            pred = F.softmax(pred, -1)
        prob, loss = self.compute_prob_and_loss(pred, y)
        return prob, loss

    def random_hidden(self, x, if_pred):
        path_random = self.E_td[x, :].sum(dim=1)
        path_random = F.leaky_relu(path_random)
        path_random = self.dropout(path_random)
        path_random = path_random.max(dim=0)[0]
        # path_random = self.random_layer(path_random)
        # path_random = F.leaky_relu(path_random)
        if if_pred:
            path_random = self.w_random.mul(path_random).sum(1) + self.b_random
            path_random = F.softmax(path_random, dim=-1)
        return path_random

    def wae_hidden(self, x, if_pred):
        path_wae = self.wae_encoder(x)
        path_wae = F.leaky_relu(path_wae)
        path_wae = path_wae.max(dim=0)[0]
        # path_wae = self.wae_layer(path_wae)
        # path_wae = F.leaky_relu(path_wae)
        if if_pred:
            path_wae = self.w_wae.mul(path_wae).sum(1) + self.b_wae
            path_wae = F.softmax(path_wae, dim=-1)
        return path_wae

    # 根据softmax后得到的概率值计算loss
    def predAndLoss(self, pred, ylabel):
        loss = (ylabel - pred).pow(2).sum()
        return pred, loss

    # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
    def compute_prob_and_loss(self, pred, label):
        label_num = label.cpu().numpy().tolist().index(1)
        l = torch.Tensor([label_num]).cuda(device).long()
        loss = self._loss(pred.unsqueeze(0), l)
        # prob = F.softmax(pred, -1)
        prob = pred
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x_random, x_response):
        if x_random.dim() == 1: x = torch.unsqueeze(x_random, 0)
        if x_response.dim() == 1: x = torch.unsqueeze(x_response, 0)
        # final_state = self.compute_tree_states(x)
        if _args.novae:
            pred = self.random_hidden(x_random, if_pred=True)
        elif _args.norandom:
            pred = self.wae_hidden(x_response, if_pred=True)
        else:
            random_hidden = self.random_hidden(x_random, if_pred=False)
            wae_hidden = self.wae_hidden(x_response, if_pred=False)
            pred = torch.cat([random_hidden, wae_hidden], -1)
            pred = self.w_cat.mul(pred).sum(1) + self.b_cat
            # 直接相加，无softmax：0.6926
            # 相加+softmax：0.6588
            pred = F.softmax(pred, -1)
        return pred


class PathWAEOld(nn.Module):
    def __init__(self, word_dim, wae_weight_path,
                 random_dim=100, wae_dim=50,
                 class_num=4, momentum=0.9, ):
        super(PathWAEOld, self).__init__()
        self.word_dim = word_dim
        self.random_dim = random_dim
        self.wae_dim = wae_dim

        self.class_num = class_num
        self.momentum = momentum
        self.device = _args.cuda
        self.wae_weight_path = wae_weight_path
        self.freeze_wae = _args.freeze_wae
        self.wae_dropout = _args.wae_dropout

        self.E_td = nn.parameter.Parameter(self.init_matrix([self.word_dim, self.random_dim]))
        # self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim + self.random_dim]))
        # self.b_out_td = nn.parameter.Parameter(self.init_vector([self.class_num]))

        self.wae_encoder = WAEEncoder(self.wae_weight_path, _args.vocab_dim,
                                      self.device, self.freeze_wae, self.wae_dropout)
        self.dropout = torch.nn.Dropout(0.3)
        # self.w_random = nn.parameter.Parameter(self.init_matrix([self.random_dim, self.random_dim]))
        # self.b_random = nn.parameter.Parameter(self.init_matrix([self.random_dim]))
        # self.w_wae = nn.parameter.Parameter(self.init_matrix([self.wae_dim, 50]))
        # self.b_wae = nn.parameter.Parameter(self.init_matrix([50]))
        if _args.novae:
            self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
        elif _args.norandom:
            self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
        else:
            # self.w_mid = nn.parameter.Parameter(
            #     self.init_matrix([(self.wae_dim + self.random_dim) // 2, self.wae_dim + self.random_dim]))
            # self.b_mid = nn.parameter.Parameter(self.init_matrix([(self.wae_dim + self.random_dim) // 2]))
            # self.w_out = nn.parameter.Parameter(
            #     self.init_matrix([self.class_num, (self.wae_dim + self.random_dim) // 2]))
            self.w_out = nn.parameter.Parameter(
                self.init_matrix([self.class_num, self.wae_dim + self.random_dim]))
        self.b_out = nn.parameter.Parameter(self.init_matrix([self.class_num]))
        self.relu = torch.nn.ReLU()
        self._loss = torch.nn.CrossEntropyLoss()
        self.layer_norm = nn.LayerNorm(self.wae_dim + self.random_dim)

    def forward(self, x, y):
        if x.dim() == 1: x = torch.unsqueeze(x, 0)
        final_state = self.compute_tree_states(x)
        prob, loss = self.compute_prob_and_loss(final_state, y)
        return prob, loss

    def compute_tree_states(self, x):
        # 计算root的隐含状态
        path_random = self.E_td[x, :].sum(dim=1)
        if _args.novae:
            path_hidden = path_random
        else:
            path_wae = self.wae_encoder(x)
            # path_wae = path_random
            if _args.norandom:
                path_hidden = path_wae
            else:
                path_hidden = torch.cat([path_random, path_wae], -1)
                # path_hidden = path_random + path_wae
        path_hidden = F.leaky_relu(path_hidden)
        path_hidden = self.dropout(path_hidden)
        path_max = path_hidden.max(dim=0)[0]
        # path_min = path_hidden.min(dim=0)[0]
        path_hidden = path_max
        # path_hidden = torch.cat([path_max, path_min], -1)
        # path_hidden = self.layer_norm(path_hidden)
        # path_hidden = self.w_mid.mul(path_hidden).sum(1) + self.b_mid
        path_hidden = self.w_out.mul(path_hidden).sum(1) + self.b_out
        path_hidden = F.softmax(path_hidden, dim=-1)
        return path_hidden

    # 根据softmax后得到的概率值计算loss
    def predAndLoss(self, pred, ylabel):
        loss = (ylabel - pred).pow(2).sum()
        return pred, loss

    # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
    def compute_prob_and_loss(self, pred, label):
        label_num = label.cpu().numpy().tolist().index(1)
        l = torch.Tensor([label_num]).cuda(device).long()
        loss = self._loss(pred.unsqueeze(0), l)
        # prob = F.softmax(pred, -1)
        prob = pred
        return prob, loss

    def init_vector(self, shape):
        return torch.zeros(shape, device=device)

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))

    def predict_up(self, x):
        if x.dim() == 1: x = torch.unsqueeze(x, 0)
        final_state = self.compute_tree_states(x)
        # final_state = F.softmax(final_state, -1)
        return final_state

# class PathVoting(nn.Module):
#     def __init__(self, word_dim, random_dim=100,
#                  vae_dim=81, class_num=4,
#                  degree=2, momentum=0.9,
#                  trainable_embeddings=True,
#                  labels_on_nonroot_nodes=False,
#                  irregular_tree=True,
#                  ):
#         super(PathVoting, self).__init__()
#         self.word_dim = word_dim
#         self.random_dim = random_dim
#         self.vae_dim = vae_dim
#
#         self.class_num = class_num
#         self.momentum = momentum
#         self.irregular_tree = irregular_tree
#         self.device = _args.cuda
#
#         self.E_td = nn.parameter.Parameter(self.init_matrix([self.word_dim, self.random_dim]))
#         self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
#         # self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.random_dim + self.vae_dim]))
#         self.b_out_td = nn.parameter.Parameter(self.init_vector([self.class_num]))
#
#         self._vae = PretrainedVAE(model_archive=vampire_pre_model_path,
#                                   device=self.device,
#                                   background_frequency=vampire_bgfreq_path,
#                                   requires_grad=False,
#                                   scalar_mix=None,
#                                   dropout=0.5)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.w_random = nn.parameter.Parameter(self.init_matrix([self.random_dim, self.random_dim]))
#         self.b_random = nn.parameter.Parameter(self.init_matrix([self.random_dim]))
#         self.w_vae = nn.parameter.Parameter(self.init_matrix([self.vae_dim, 50]))
#         self.b_vae = nn.parameter.Parameter(self.init_matrix([50]))
#         if _args.novae:
#             self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
#         else:
#             self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
#         self.b_out = nn.parameter.Parameter(self.init_matrix([self.class_num]))
#         self.relu = torch.nn.ReLU()
#         self._loss = torch.nn.CrossEntropyLoss()
#
#     def forward(self, x, y):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         final_state = self.compute_tree_states(x)
#         # prob, loss = self.predAndLoss(final_state, y)
#         prob, loss = self.compute_prob_and_loss(final_state, y)
#         return prob, loss
#
#     def compute_tree_states(self, x):
#         # 计算root的隐含状态
#         path_random = self.E_td[x, :].sum(dim=1)
#         # path_random = self.E_td[x, :]
#         # path_random = torch.mm(path_random, self.w_random) + self.b_random
#         if _args.novae:
#             path_hidden = path_random
#         else:
#             path_vae = self._vae(x)
#             # path_vae = (path_vae.unsqueeze(0)
#             #             .expand(x.shape[1], x.shape[0], -1)
#             #             .permute(1, 0, 2).contiguous())
#             # path_hidden = path_vae
#             # path_vae = torch.mm(path_vae, self.w_vae) + self.b_vae
#             path_hidden = torch.cat([path_random, path_vae], -1)
#         # (P,D181)*(D181,C4)=>(P,C4)
#         path_hidden = self.dropout(path_hidden)
#         # path_hidden = path_hidden.sum(1)
#         # path_hidden = path_hidden / (x.shape[0] * 0.5)
#         path_hidden = path_hidden.max(dim=0)[0]
#         # path_hidden = self.dropout(path_hidden)
#         path_hidden = self.w_out.mul(path_hidden).sum(dim=1) + self.b_out
#         path_hidden = F.softmax(path_hidden, dim=-1)
#         return path_hidden
#
#     # 分开计算random和vae，然后concat并mlp
#     def compute_path_hidden(self, x):
#         path_random = self.E_td[x, :].sum(dim=1)
#         path_vae = self._vae(x)
#         vae_hidden = self.w_vae.mul(path_vae).sum(dim=1) + self.b_vae
#         path_hidden = torch.concat([vae_hidden, path_random])
#         path_hidden = self.dropout(path_hidden)
#         path_hidden = self.w_out.mul(path_hidden).sum(dim=1) + self.b_out
#         # path_hidden = F.softmax(path_hidden, dim=-1)
#         return path_hidden
#
#     # 根据softmax后得到的概率值计算loss
#     def predAndLoss(self, pred, ylabel):
#         loss = (ylabel - pred).pow(2).sum()
#         return pred, loss
#
#     # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
#     def compute_prob_and_loss(self, pred, label):
#         label_num = label.cpu().numpy().tolist().index(1)
#         l = torch.Tensor([label_num]).cuda(device).long()
#         loss = self._loss(pred.unsqueeze(0), l)
#         prob = F.softmax(pred, -1)
#         return prob, loss
#
#     def init_vector(self, shape):
#         return torch.zeros(shape, device=device)
#
#     def init_matrix(self, shape):
#         return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))
#
#     def predict_up(self, x):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         final_state = self.compute_tree_states(x)
#         final_state = F.softmax(final_state, -1)
#         return final_state
#
#
# # path和vae分别mlp再concat
# class PathVAE(nn.Module):
#     def __init__(self, word_dim, random_dim=100,
#                  vae_dim=81, class_num=4,
#                  momentum=0.9,
#                  irregular_tree=True,
#                  ):
#         super(PathVAE, self).__init__()
#         self.word_dim = word_dim
#         self.random_dim = random_dim
#         self.vae_dim = vae_dim
#
#         self.class_num = class_num
#         self.momentum = momentum
#         self.irregular_tree = irregular_tree
#         self.device = _args.cuda
#
#         self.E_td = nn.parameter.Parameter(self.init_matrix([self.word_dim, self.random_dim]))
#         self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
#         # self.W_out_td = nn.parameter.Parameter(self.init_matrix([self.Nclass, self.random_dim + self.vae_dim]))
#         self.b_out_td = nn.parameter.Parameter(self.init_vector([self.class_num]))
#
#         self._vae = PretrainedVAE(model_archive=vampire_pre_model_path,
#                                   device=self.device,
#                                   background_frequency=vampire_bgfreq_path,
#                                   requires_grad=False,
#                                   scalar_mix=None,
#                                   dropout=0.2)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.w_random = nn.parameter.Parameter(self.init_matrix([self.random_dim, self.random_dim]))
#         self.b_random = nn.parameter.Parameter(self.init_matrix([self.random_dim]))
#         self.w_vae = nn.parameter.Parameter(self.init_matrix([self.vae_dim, 50]))
#         self.b_vae = nn.parameter.Parameter(self.init_matrix([50]))
#         if _args.novae:
#             self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
#         else:
#             self.w_out = nn.parameter.Parameter(self.init_matrix([self.class_num, self.vae_dim + self.random_dim]))
#         self.b_out = nn.parameter.Parameter(self.init_matrix([self.class_num]))
#         self.relu = torch.nn.ReLU()
#         self._loss = torch.nn.CrossEntropyLoss()
#
#     def forward(self, x, y):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         final_state = self.compute_tree_states(x)
#         # prob, loss = self.predAndLoss(final_state, y)
#         prob, loss = self.compute_prob_and_loss(final_state, y)
#         return prob, loss
#
#     def compute_tree_states(self, x):
#         # 计算root的隐含状态
#         path_random = self.E_td[x, :].sum(dim=1)
#         # path_random = self.E_td[x, :]
#         # path_random = torch.mm(path_random, self.w_random) + self.b_random
#         if _args.novae:
#             path_hidden = path_random
#         else:
#             path_vae = self._vae(x)
#             # path_vae = (path_vae.unsqueeze(0)
#             #             .expand(x.shape[1], x.shape[0], -1)
#             #             .permute(1, 0, 2).contiguous())
#             # path_hidden = path_vae
#             # path_vae = torch.mm(path_vae, self.w_vae) + self.b_vae
#             path_hidden = torch.cat([path_random, path_vae], -1)
#         # (P,D181)*(D181,C4)=>(P,C4)
#         path_hidden = self.dropout(path_hidden)
#         # path_hidden = path_hidden.sum(1)
#         # path_hidden = path_hidden / (x.shape[0] * 0.5)
#         path_hidden = path_hidden.max(dim=0)[0]
#         # path_hidden = self.dropout(path_hidden)
#         path_hidden = self.w_out.mul(path_hidden).sum(dim=1) + self.b_out
#         path_hidden = F.softmax(path_hidden, dim=-1)
#         return path_hidden
#
#     # 分开计算random和vae，然后concat并mlp
#     def compute_path_hidden(self, x):
#         path_random = self.E_td[x, :].sum(dim=1)
#         path_vae = self._vae(x)
#         vae_hidden = self.w_vae.mul(path_vae).sum(dim=1) + self.b_vae
#         path_hidden = torch.concat([vae_hidden, path_random])
#         path_hidden = self.dropout(path_hidden)
#         path_hidden = self.w_out.mul(path_hidden).sum(dim=1) + self.b_out
#         # path_hidden = F.softmax(path_hidden, dim=-1)
#         return path_hidden
#
#     # 根据softmax后得到的概率值计算loss
#     def predAndLoss(self, pred, ylabel):
#         loss = (ylabel - pred).pow(2).sum()
#         return pred, loss
#
#     # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
#     def compute_prob_and_loss(self, pred, label):
#         label_num = label.cpu().numpy().tolist().index(1)
#         l = torch.Tensor([label_num]).cuda(device).long()
#         loss = self._loss(pred.unsqueeze(0), l)
#         prob = F.softmax(pred, -1)
#         return prob, loss
#
#     def init_vector(self, shape):
#         return torch.zeros(shape, device=device)
#
#     def init_matrix(self, shape):
#         return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))
#
#     def predict_up(self, x):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         final_state = self.compute_tree_states(x)
#         final_state = F.softmax(final_state, -1)
#         return final_state

# class PathWAE(nn.Module):
#     def __init__(self, word_dim, wae_weight_path,
#                  random_dim=100, wae_dim=50,
#                  class_num=4, momentum=0.9, ):
#         super(PathWAE, self).__init__()
#         self.word_dim = word_dim
#         self.random_dim = random_dim
#         self.wae_dim = wae_dim
#
#         self.class_num = class_num
#         self.momentum = momentum
#         self.device = _args.cuda
#         self.wae_weight_path = wae_weight_path
#         self.freeze_wae = _args.freeze_wae
#         self.wae_dropout = _args.wae_dropout
#         self.wae_encoder = WAEEncoder(self.wae_weight_path, _args.vocab_dim,
#                                       self.device, self.freeze_wae, self.wae_dropout)
#
#         self.E_td = nn.parameter.Parameter(self.init_matrix([self.word_dim, self.random_dim]))
#         self.w_random = nn.parameter.Parameter(self.init_matrix([self.class_num, self.random_dim]))
#         self.b_random = nn.parameter.Parameter(self.init_matrix([self.class_num]))
#         self.w_wae = nn.parameter.Parameter(self.init_matrix([self.class_num, self.wae_dim]))
#         self.b_wae = nn.parameter.Parameter(self.init_matrix([self.class_num]))
#
#         self.random_layer = nn.Linear(self.random_dim, self.class_num)
#         self.wae_layer = nn.Linear(self.wae_dim, self.class_num)
#
#         self.w_cat = nn.parameter.Parameter(self.init_matrix([self.class_num, self.class_num * 2]))
#         self.b_cat = nn.Parameter(self.init_matrix([self.class_num]))
#
#         self.dropout = torch.nn.Dropout(0.3)
#         self.dropout1 = torch.nn.Dropout(0.1)
#         self.relu = torch.nn.ReLU()
#         self._loss = torch.nn.CrossEntropyLoss()
#         self.layer_norm = nn.LayerNorm(self.wae_dim + self.random_dim)
#
#     def forward(self, x, y):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         if _args.novae:
#             # 0.7027
#             pred = self.random_predict(x)
#         elif _args.norandom:
#             # 0.6588
#             pred = self.wae_predict(x)
#         else:
#             random_pred = self.random_predict(x)
#             wae_pred = self.wae_predict(x)
#             pred = wae_pred + random_pred
#             # 直接相加，无softmax：0.6926
#             # pred = torch.cat([random_pred, wae_pred], -1)
#             # pred = self.dropout1(pred)
#             # pred = self.w_cat.mul(pred).sum(1) + self.b_cat
#             # pred = self.dropout1(pred)
#             # 相加+softmax：0.6588
#             # pred = F.softmax(pred, -1)
#         prob, loss = self.compute_prob_and_loss(pred, y)
#         return prob, loss
#
#     def random_predict(self, x):
#         path_random = self.E_td[x, :].sum(dim=1)
#         path_random = F.leaky_relu(path_random)
#         path_random = self.dropout(path_random)
#         path_random = path_random.max(dim=0)[0]
#         # path_random = self.random_layer(path_random)
#         # path_random = F.leaky_relu(path_random)
#         path_random = self.w_random.mul(path_random).sum(1) + self.b_random
#         path_random = F.softmax(path_random, dim=-1)
#         return path_random
#
#     def wae_predict(self, x):
#         path_wae = self.wae_encoder(x)
#         path_wae = F.leaky_relu(path_wae)
#         path_wae = path_wae.max(dim=0)[0]
#         # path_wae = self.wae_layer(path_wae)
#         # path_wae = F.leaky_relu(path_wae)
#         path_wae = self.w_wae.mul(path_wae).sum(1) + self.b_wae
#         path_wae = F.softmax(path_wae, dim=-1)
#         return path_wae
#
#     # 根据softmax后得到的概率值计算loss
#     def predAndLoss(self, pred, ylabel):
#         loss = (ylabel - pred).pow(2).sum()
#         return pred, loss
#
#     # 根据预测值计算每一类别的计算概率；根据标签和预测值计算损失
#     def compute_prob_and_loss(self, pred, label):
#         label_num = label.cpu().numpy().tolist().index(1)
#         l = torch.Tensor([label_num]).cuda(device).long()
#         loss = self._loss(pred.unsqueeze(0), l)
#         # prob = F.softmax(pred, -1)
#         prob = pred
#         return prob, loss
#
#     def init_vector(self, shape):
#         return torch.zeros(shape, device=device)
#
#     def init_matrix(self, shape):
#         return torch.from_numpy(np.random.normal(scale=0.1, size=shape).astype('float32'))
#
#     def predict_up(self, x):
#         if x.dim() == 1: x = torch.unsqueeze(x, 0)
#         # final_state = self.compute_tree_states(x)
#         if _args.novae:
#             pred = self.random_predict(x)
#         elif _args.norandom:
#             pred = self.wae_predict(x)
#         else:
#             random_pred = self.random_predict(x)
#             wae_pred = self.wae_predict(x)
#             # pred = torch.cat([random_pred, wae_pred], -1)
#             # pred = self.w_cat.mul(pred).sum(1) + self.b_cat
#             # 直接相加，无softmax：0.6926
#             pred = random_pred + wae_pred
#             # 相加+softmax：0.6588
#             # pred = F.softmax(pred, -1)
#         # final_state = F.softmax(final_state, -1)
#         return pred
