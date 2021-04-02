import os
import torch
from os.path import join
from get_args import _args as _args

device = torch.device("cuda:%d" % _args.cuda
                      if torch.cuda.is_available()
                         and _args.cuda >= 0 else "cpu")

# 项目目录
project_dir_path = os.path.dirname(__file__)
# 数据文件夹路径
data_dir_path = join(project_dir_path, 'data')
# early detection预处理生成的文件所在文件夹
early_detection_dir_path = join(data_dir_path, "early_detection_pre")
if not os.path.exists(early_detection_dir_path):
    os.mkdir(early_detection_dir_path)

# 训练集和测试集路径
fold_dir_path = join(data_dir_path, 'nfold')
train_path = join(fold_dir_path, 'Train%sFold%d.txt' % (_args.dataset, _args.fold))
test_path = join(fold_dir_path, 'Test%sFold%d.txt' % (_args.dataset, _args.fold))
# t15,t16的label文件
label_path = join(data_dir_path, 'label%s.txt' % _args.dataset)

# 树文件夹路径;每棵树中每行是：父子id对以及对应的时间
tree_dir_path = join(data_dir_path, 'Tree%s' % _args.dataset)
# source 推特路径
source_path = join(data_dir_path, 'source_tweets%s.txt' % _args.dataset)
# response 推特路径
response_path = join(data_dir_path, 'tweet_response%s_clean.txt' % _args.dataset)
# 全部twitter 文本对应的文件路径：[text+id,...];可用于根据id获取text
text_all_path = join(data_dir_path, 'tweet1516_all.txt')

# 全部路径节点的合计
path_ids_path = join(data_dir_path, 'path%s_ids.txt' % _args.dataset)
# 全部路径id：text的原始文本
path_texts_raw_path = join(data_dir_path, 'path%s_texts_raw.txt' % _args.dataset)
# 全部路径文本和标签的合计
path_texts_path = join(data_dir_path, 'path%s_texts.txt' % _args.dataset)
# 回复文本构成的路径文本路径；删除了源文本
path_response_text_raw_path = join(data_dir_path, 'path%s_response_texts_raw.txt' % _args.dataset)
# 预处理之后的回复路径文本：表情替换，删除@和url等
path_response_text_path = join(data_dir_path, 'path%s_response_texts.txt' % _args.dataset)

# 测试用：合并twitter15,16用于无监督训练
path_response15_text_path = join(data_dir_path, 'path15_response_texts.txt')
path_response16_text_path = join(data_dir_path, 'path16_response_texts.txt')

#######################################
# 为WAE的无监督训练预处理数据
# 保存全部预处理数据的文件夹
path_vocab4random = join(data_dir_path, 'path%svocab4random' % _args.dataset)
response_path_vocab4wae = join(data_dir_path, 'response_path%svocab4wae' % _args.dataset)

# random对应的npz文件和中间文件
path_random_npz = join(path_vocab4random, 'all.npz')
path_random_id2paths_dict_path = join(path_vocab4random, 'id2paths%s_dict.json' % _args.dataset)
early_path_random_id2paths_dict_path = join(path_vocab4random,
                                            'early%d_id2paths%s_dict.json' % (_args.interval, _args.dataset))

path_bert_id2paths_dict_path = join(path_vocab4random, 'id2paths%s_bert_dict.json' % _args.dataset)

# wae对应的npz文件和中间文件
response_wae_npz = join(response_path_vocab4wae, "all.npz")
response_id2paths_dict_path = join(response_path_vocab4wae, 'id2paths%s_dict.json' % _args.dataset)
early_response_id2paths_dict_path = join(response_path_vocab4wae,
                                         'early%d_id2paths%s_dict.json' % (_args.interval, _args.dataset))

# wae-encoder的权重保存文件夹
wae_weight_dir_path = join(data_dir_path, 'wae_weight', "path5000")
# wae_weight_dir_path = "/data1/zperData/FakeNews/w-lda-master/examples/results/path16_5000/2019-12-29-19-54-20/weights/encoder/"
wae_best_encoder_path = join(wae_weight_dir_path, 'Enc' + _args.dataset)

# early detection中添加的：根据每个tweet id得到对应的文本token ids的字典
random_tweet2token_ids_dict_path = join(data_dir_path, "early_random%s_ids.json" % _args.dataset)
response_tweet2token_ids_dict_path = join(data_dir_path, "early_response%s_ids.json" % _args.dataset)

# early detection中需要的：每棵树对应的路径的node-id表示的词典
path_node_ids_dict_path = join(early_detection_dir_path,
                               "early_twitter%s_interval%d.json" % (_args.dataset, _args.interval))

#####################################################################
# 使用预训练词向量
bert_weight_type = 'bert-base-uncased'
bert_weight_dir_path = join(data_dir_path, 'bert_pre')
bert_base_weight_path = join(bert_weight_dir_path, 'bert-base-uncased-pytorch_model.bin')

# OSError: libcublas.so.10.0: cannot open shared object file: No such file or directory
# LD_LIBRARY_PATH
# /usr/local/cuda-10.0/lib64/
