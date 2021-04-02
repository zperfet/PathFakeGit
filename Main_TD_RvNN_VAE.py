import sys
from models import TD_RvNN_VAE
import time
import random
from torch import optim
import datetime
from evaluate import *
from VAE.random_search import *
from config import *
import os
from data_io import get_index_dict, loadLabel, load_sparse
from get_args import _ma_vae_args
from base import pad_zero


# 将索引:词频对应的str转换为词索引向量和词频向量
def str2matrix(Str, MaxL):
    word_freq, word_index = [], []
    l = 0
    for pair in Str.split(' '):
        word_freq.append(float(pair.split(':')[1]))
        word_index.append(int(pair.split(':')[0]))
        l += 1
    ladd = [0] * (MaxL - l)
    word_freq += ladd
    word_index += ladd
    return word_freq, word_index


# 获取树的叶子节点的node数
# 删除全部出现的父节点，剩下的就是叶子结点
def get_leaf_nodes(edge):
    nodes = list(range(len(edge)))
    for i in edge:
        if i[0] in nodes:
            nodes.remove(i[0])
    return nodes


def loadData(train_path, test_path, label_path, index_text_path,
             tree_dir_path, source_path, response_path, npz_path):
    # 从本地加载标签
    label_dic = {}
    for line in open(label_path):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label_dic[eid] = label.lower()
    print("loading tree label:", len(label_dic))

    # 从本地加载id-text字典
    # text包括source和resposne
    id_text_dict = {}
    with open(response_path, 'r', encoding='utf-8')as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            id_text_dict[line[0]] = line[1]
    with open(source_path, 'r', encoding='utf-8')as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            id_text_dict[line[0]] = line[1]
    print('load %d text(source && response)' % len(id_text_dict))

    # node_dic中保存每个节点的text信息：[token_id,...]
    # edge_dic中保存节点之间的连接关系；通过根节点和连接关系可以得到树结构
    node_dic = {}
    edge_dic = {}
    # 全部文件名
    names = os.listdir(tree_dir_path)
    # dict[id:index],根据id得到在npz矩阵中的index
    id_index_dict = get_index_dict(index_text_path)
    # 从本地加载npz文件，包含了全部text的count编码
    # npz由vampire预处理得到，简化了分词等预处理过程，直接由id得到vector
    mat = load_sparse(npz_path)
    # 处理全部树文件
    for name_cnt, name in enumerate(names):
        if (name_cnt + 1) % 100 == 0:
            print('处理第%d个文件' % (name_cnt + 1))
        source_id = name.split('.')[0]
        # 文件路径
        path = join(tree15_dir_path, name)
        # 初始化当前id的dict_val:[]
        node_dic[source_id] = []
        # 根据id得到在npz文件中的index
        text_index = id_index_dict[source_id]
        # 获取文本的id表示：mat矩阵中index行中非零元素的下标；不考虑顺序
        vector_text = mat[text_index].nonzero()[1]
        node_dic[source_id].append(vector_text.tolist())
        # 记录树中的边，[父节点id，子节点id]
        edge_dic[source_id] = []
        edge_dic[source_id].append([-1, 0])
        with open(path, 'r')as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            # 统计当前所有的id，并赋予node_index值(0,1,2...)，构建dict{id:node_index}
            # 根据id获取node数
            id2node_dict = {}
            id2node_dict[source_id] = 0
            max_len = len(vector_text)
            # line:[父节点id，父节点time，子节点id，子节点time]
            for line in lines:
                # 如果没有子节点，直接跳过；小部分节点直接为空，即树中只有一个根节点
                if not line[0]: break
                # r如果子节点是根节点，则直接跳过这一条边
                if line[2] == source_id:
                    continue
                # 如果父节点或者子节点还不存在id2node_dict中，则添加新的节点
                if line[0] not in id2node_dict.keys():
                    id2node_dict[line[0]] = len(id2node_dict)
                if line[2] not in id2node_dict.keys():
                    id2node_dict[line[2]] = len(id2node_dict)
                text_index = id_index_dict[line[2]]
                vector_text = mat[text_index].nonzero()[1]
                node_dic[source_id].append(vector_text.tolist())
                # node_dic[source_id].append(tokens_id)
                edge_dic[source_id].append([id2node_dict[line[0]], id2node_dict[line[2]]])
                # 记录文本的最大长度
                max_len = max(len(vector_text), max_len)
            # pad 0
            node_dic[source_id] = pad_zero(node_dic[source_id], max_len)
    print('read tree no:', len(node_dic))

    tree_train, edge_train, y_train, leaf_idxs_train, c = [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(train_path):
        # if c > 8: break
        eid = eid.rstrip()
        if not label_dic.__contains__(eid): continue
        if not node_dic.__contains__(eid): continue
        if len(node_dic[eid]) <= 0:
            continue
        label = label_dic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        tree = node_dic[eid]
        edge = edge_dic[eid]
        tree_train.append(tree)
        edge_train.append(edge)
        # 获取叶子节点的node数
        leaf_idxs = get_leaf_nodes(edge)
        leaf_idxs_train.append(leaf_idxs)
        c += 1
    print("loading train set:", l1, l2, l3, l4)

    tree_test, edge_test, y_test, leaf_idxs_test, c = [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(test_path):
        # if c > 4: break
        eid = eid.rstrip()
        if not label_dic.__contains__(eid): continue
        if not node_dic.__contains__(eid): continue
        if len(node_dic[eid]) <= 0:
            continue
            # 1. load label
        label = label_dic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        tree = node_dic[eid]
        edge = edge_dic[eid]
        tree_test.append(tree)
        edge_test.append(edge)
        # 获取叶子节点的node数
        leaf_idxs = get_leaf_nodes(edge)
        leaf_idxs_test.append(leaf_idxs)
        c += 1
    print("loading test set:", l1, l2, l3, l4)
    return tree_train, edge_train, y_train, leaf_idxs_train, \
           tree_test, edge_test, y_test, leaf_idxs_test


# 预处理数据
print("Step1:processing data")
tree_train, edge_train, y_train, leaf_idxs_train, \
tree_test, edge_test, y_test, leaf_idxs_test = \
    loadData(train_path, test_path, label15_path, text_all_path,
             tree15_dir_path, source15_path, response15_path, Ma_VAE_NPZ)
print()

# 定义模型
print('Step2:build model')
t0 = time.time()
model = TD_RvNN_VAE.RvNN(_ma_vae_args.vocab_dim, _ma_vae_args.random_dim, _ma_vae_args.class_num)
model.to(device)
t1 = time.time()
print('Recursive model established,', (t1 - t0) / 60, 's\n')

# 3. looping SGD
print('Step3:start training')
optimizer = optim.Adagrad(model.parameters(), lr=_ma_vae_args.lr)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.2, 2)
losses_5, losses = [], []
num_examples_seen = 0
indexs = list(range(len(y_train)))
highest_acc = 0
for epoch in range(1, _ma_vae_args.epoch + 1):
    # one SGD
    random.shuffle(indexs)
    for cnt, i in enumerate(indexs):
        pred_y, loss = model.forward(torch.Tensor(tree_train[i]).cuda(device).long(),
                                     torch.LongTensor(edge_train[i]).cuda(device).long(),
                                     torch.Tensor(leaf_idxs_train[i]).cuda(device).long(),
                                     torch.Tensor(y_train[i]).cuda(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().tolist())
        num_examples_seen += 1
        if (cnt + 1) % 100 == 0:
            # print("iteration:%d/%d" % (cnt, len(indexs)))
            break
        # print("epoch=%d: idx=%d, loss=%f" % (epoch, i, np.mean(losses)))
    # cal loss & evaluate
    if epoch % 1 == 0:
        losses_5.append((num_examples_seen, np.mean(losses)))
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
        sys.stdout.flush()
        prediction = []
        for j in range(len(y_test)):
            prediction.append(
                model.predict_up(torch.Tensor(tree_test[j]).cuda(device).long(),
                                 torch.Tensor(edge_test[j]).cuda(device).long(),
                                 torch.Tensor(leaf_idxs_test[j]).cuda(device).long())
                    .cpu().data.numpy().tolist())
        # print("predictions:", prediction)
        res = evaluation_4class(prediction, y_test)
        highest_acc = max(highest_acc, res[1])
        # print('results:', res)
        # print()
        sys.stdout.flush()
        # Adjust the learning rate if loss increases
        # if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
        #     lr = lr * 0.5
        #     print("Setting learning rate to %.12f" % lr)
        #     sys.stdout.flush()
    sys.stdout.flush()
    losses = []
print('最高acc：', highest_acc)
