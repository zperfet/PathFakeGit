# 基于PyTorch复现并修正后的马金TD方法
# 修正：叶子节点选择错误；SGD->Adagrad
import sys
import numpy as np
from models import TD_RvNN_Ma
import time
import random
from torch import optim
import datetime
from evaluate import *
from config import *
from data_io import loadLabel
from get_args import _ma_args


# str = index:wordfreq index:wordfreq
def str2matrix(Str, MaxL):
    word_freq, word_index = [], []
    l = 0
    for pair in Str.split(' '):
        word_freq.append(float(pair.split(':')[1]))
        word_index.append(int(pair.split(':')[0]))
        l += 1
    l_add = [0] * (MaxL - l)
    word_freq += l_add
    word_index += l_add
    return word_freq, word_index


def constructTree(tree):
    # tree: {index1:{'parent':, 'maxL':, 'vec':}
    # 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN_Ma.Node_tweet(idx=i)
        index2node[i] = node
    # 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'], tree[j]['maxL'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        # nodeC.time = tree[j]['post_t']
        # not root node
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        # root node
        else:
            root = nodeC
    # 3. convert tree to DNN input
    parent_num = tree[j]['parent_num']
    ini_x, ini_index = str2matrix("0:0", tree[j]['maxL'])
    x_word, x_index, tree, leaf_idxs = TD_RvNN_Ma.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, leaf_idxs


################################# loas data ###################################
def loadData(tree_path, train_path, test_path, label_path, train_threshold):
    labelDic = {}
    for line in open(label_path):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
    print("loading tree label:", len(labelDic))

    treeDic = {}
    for line in open(tree_path):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
        Vec = line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'parent_num': parent_num, 'maxL': maxL, 'vec': Vec}
    print("reading tree no:", len(treeDic))

    tree_train, word_train, index_train, y_train, leaf_idxs_train, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    with open(train_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        random.shuffle(train_ids)
        for eid in train_ids[:train_threshold]:
            # if c > 8: break
            eid = eid.rstrip()
            if not labelDic.__contains__(eid): continue
            if not treeDic.__contains__(eid): continue
            if len(treeDic[eid]) <= 0:
                continue
            # 1. load label
            label = labelDic[eid]
            y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
            y_train.append(y)
            # 2. construct tree
            x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
            tree_train.append(tree)
            word_train.append(x_word)
            index_train.append(x_index)
            leaf_idxs_train.append(leaf_idxs)
            c += 1
    print("loading train set:", l1, l2, l3, l4)

    tree_test, word_test, index_test, leaf_idxs_test, y_test, c = [], [], [], [], [], 0
    l1, l2, l3, l4 = 0, 0, 0, 0
    for eid in open(test_path):
        # if c > 4: break
        eid = eid.rstrip()
        if not labelDic.__contains__(eid): continue
        if not treeDic.__contains__(eid): continue
        if len(treeDic[eid]) <= 0:
            continue
            # 1. load label
        label = labelDic[eid]
        y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        # 2. construct tree
        x_word, x_index, tree, leaf_idxs = constructTree(treeDic[eid])
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        leaf_idxs_test.append(leaf_idxs)
        c += 1
    print("loading test set:", l1, l2, l3, l4)

    print("train no:", len(tree_train), len(word_train), len(index_train), len(leaf_idxs_train), len(y_train))
    print("test no:", len(tree_test), len(word_test), len(index_test), len(leaf_idxs_test), len(y_test))
    return tree_train, word_train, index_train, leaf_idxs_train, y_train, tree_test, word_test, index_test, leaf_idxs_test, y_test


##################################### MAIN ####################################
# 1. load tree & word & index & label
print("Step1:processing data")
tree_train, word_train, index_train, leaf_idxs_train, y_train, \
tree_test, word_test, index_test, leaf_idxs_test, y_test = \
    loadData(TD_RvNN_TFIDF_path, train_path, test_path, label15_path, _ma_args.train_threshold)
print()

# 2. ini RNN model
print('Step2:build model')
t0 = time.time()
model = TD_RvNN_Ma.RvNN(_ma_args.vocab_dim, _ma_args.random_dim, _ma_args.class_num)
model.to(device)
t1 = time.time()
print('Recursive model established,', (t1 - t0) / 60, 's\n')

# 3. looping SGD
print('Step3:start training')
optimizer = optim.Adagrad(model.parameters(), lr=0.05)
losses_5, losses = [], []
num_examples_seen = 0
indexs = list(range(len(y_train)))
highest_acc = 0
for epoch in range(1, _ma_args.epoch + 1):
    # one SGD
    random.shuffle(indexs)
    for cnt, i in enumerate(indexs):
        pred_y, loss = model.forward(torch.Tensor(word_train[i]).cuda(device),
                                     torch.LongTensor(index_train[i]).cuda(device).long(),
                                     torch.LongTensor(tree_train[i]).cuda(device),
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
                model.predict_up(torch.Tensor(word_test[j]).cuda(device),
                                 torch.Tensor(index_test[j]).cuda(device).long(),
                                 torch.Tensor(tree_test[j]).cuda(device).long(),
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
