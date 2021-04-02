import torch.nn as nn
from config import wae_best_encoder_path
from mxnet import ndarray
import torch
import torch.nn.functional as F
from get_args import _args


class WAEEncoder(nn.Module):
    def __init__(self, wae_weight_file_path, vocab_dim, device, freeze_encoder, dropout):
        super(WAEEncoder, self).__init__()
        self.wae_weight_path = wae_weight_file_path
        self.vocab_size = vocab_dim
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.layer0 = nn.Linear(5001, 100)
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 50)
        self.dropout = torch.nn.Dropout(dropout)
        if _args.load_wae_encoder:
            self.load_wae_weight(self.layer0, self.layer1, self.layer2)

    # 从本地加载训练好的wae-encoder权重，并赋予定义好的网络
    def load_wae_weight(self, layer0, layer1, layer2):
        wae_weight = ndarray.load(self.wae_weight_path)

        weight0 = torch.Tensor(wae_weight['main.0.weight'].asnumpy())
        layer0.weight = nn.parameter.Parameter(weight0)
        # if self.freeze_encoder:
        #     layer0.weight.requires_grad = False
        bias0 = torch.Tensor(wae_weight['main.0.bias'].asnumpy())
        layer0.bias = nn.parameter.Parameter(bias0)
        # if self.freeze_encoder:
        #     layer0.bias.requires_grad = False

        weight1 = torch.Tensor(wae_weight['main.1.weight'].asnumpy())
        layer1.weight = nn.parameter.Parameter(weight1)
        # if self.freeze_encoder:
        #     layer1.weight.requires_grad = False
        bias1 = torch.Tensor(wae_weight['main.1.bias'].asnumpy())
        layer1.bias = nn.parameter.Parameter(bias1)
        # if self.freeze_encoder:
        #     layer1.bias.requires_grad = False

        weight2 = torch.Tensor(wae_weight['main.2.weight'].asnumpy())
        layer2.weight = nn.parameter.Parameter(weight2)
        # if self.freeze_encoder:
        #     layer2.weight.requires_grad = False
        bias2 = torch.Tensor(wae_weight['main.2.bias'].asnumpy())
        layer2.bias = nn.parameter.Parameter(bias2)
        # if self.freeze_encoder:
        #     layer2.bias.requires_grad = False

    def forward(self, wae_input):
        one_hot_input = torch.zeros(wae_input.shape[0], self.vocab_size, device=self.device).float()
        for i in range(wae_input.shape[0]):
            one_hot_input[i, wae_input[i]] = 1.0
        output0 = F.softplus(self.layer0(one_hot_input))
        output1 = F.softplus(self.layer1(output0))
        output2 = self.layer2(output1)
        # output = torch.cat([output0, output1, output2], -1)
        output = output2
        output = self.dropout(output)
        output = F.softmax(output, -1)
        return output

# encoder = WAEEncoder(wae_best_encoder_path)
# input = torch.Tensor(1, 30001)
# output = encoder.forward(input)
# print(output.shape)
