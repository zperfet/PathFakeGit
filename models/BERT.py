import torch.nn as nn
from config import *
from mxnet import ndarray
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel


class BERTEncoder(nn.Module):
    def __init__(self, weights_type, bert_weight_dir_path, num_labels, device):
        super(BERTEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(weights_type,
                                                       cache_dir=bert_weight_dir_path,
                                                       do_lower_case=True)
        config = BertConfig.from_pretrained(weights_type, num_labels=num_labels)
        self.model = BertModel.from_pretrained(weights_type,
                                               cache_dir=bert_weight_dir_path,
                                               config=config)
        self.device = device
        self.model.to(self.device)

    def forward(self, input):
        input_ids = self.tokenizer.encode(input)
        input_ids_array = torch.tensor([input_ids, input_ids], dtype=torch.long, device=self.device)
        input_tensors = self.model(input_ids_array)[1]
        return input_tensors


encoder = BERTEncoder('bert-base-uncased', bert_weight_dir_path, 4, device)
input_text = "Let's see all hidden-states and attentions on this text"
res = encoder.forward(input_text)
print(res)
