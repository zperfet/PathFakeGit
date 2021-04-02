from scipy import sparse
from base import *
from get_args import _args as _args
import json
import codecs
# from pytorch_transformers import BertTokenizer
from config import bert_weight_type


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


# 从本地加载vampire预处理阶段得到的稀疏矩阵
def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


# 将字典保存到本地
def save_json_dict(dict2save, save_path):
    with open(save_path, 'w')as f:
        json.dump(dict2save, f)


# 从本地加载字典
def load_dict_json(load_path):
    with open(load_path, 'r')as f:
        load_dict = json.load(f)
    return load_dict


# 返回tweet文本的id列表
def get_tweet_id_lst(tweet_path):
    with open(tweet_path, 'r', encoding='utf-8')as f:
        ids = [line.strip().split('\t')[0] for line in f.readlines()]
    return ids


# 转换t15,t16标签数据
def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
        y_train = [1, 0, 0, 0]
        l1 += 1
    if label in labelset_f:
        y_train = [0, 1, 0, 0]
        l2 += 1
    if label in labelset_t:
        y_train = [0, 0, 1, 0]
        l3 += 1
    if label in labelset_u:
        y_train = [0, 0, 0, 1]
        l4 += 1
    return y_train, l1, l2, l3, l4


# 转换Pheme标签数据
def loadLabelPheme(label, l1, l2, l3):
    labelset_f, labelset_t, labelset_u = ['false'], ['true'], ['unverified']
    if label in labelset_f:
        y_train = [1, 0, 0]
        l1 += 1
    if label in labelset_t:
        y_train = [0, 1, 0]
        l2 += 1
    if label in labelset_u:
        y_train = [0, 0, 1]
        l3 += 1
    return y_train, l1, l2, l3


# 转换Pheme标签数据
def loadLabelWeibo(label, l1, l2):
    labelset_f, labelset_t = ['false', '0'], ['true', '1']
    if label in labelset_f:
        y_train = [1, 0]
        l1 += 1
    if label in labelset_t:
        y_train = [0, 1]
        l2 += 1
    return y_train, l1, l2


# 从本地加载label字典；根据id获取对应的标签
def get_label_dict(label_path):
    with open(label_path, 'r')as f:
        lines = [line.strip() for line in f.readlines()]
    id_label_dict = dict()
    for line in lines:
        line = line.split('\t')
        id, label = line[0], line[1]
        id_label_dict[id] = label
    return id_label_dict


# 从本地加载文本词典；根据id获取对应的text；这里的text对应source|response推特
def get_text_dict(text_path):
    # with open(source15_path, 'r', encoding='utf-8')as f:
    #     lines1 = [line.strip() for line in f.readlines()]
    # with open(response15_path, 'r', encoding='utf-8')as f:
    #     lines2 = [line.strip() for line in f.readlines()]
    # lines = lines1 + lines2
    with open(text_path, 'r', encoding='utf-8')as f:
        lines = [line.strip() for line in f.readlines()]
    id_text_dict = dict()
    for line in lines:
        line = line.split('\t')
        if len(line) == 1: line.append(' ')
        id, text = line[0], line[1]
        id_text_dict[id] = text
    return id_text_dict


# 从本地加载路径数据
# 参数：训练集路径，测试集路径，保存中间结果字典的路径
def load_path_data(train_data_path, test_data_path, label_path, tmp_dict_save_path, npz_path):
    # 获取标签字典
    label_dic = get_label_dict(label_path)
    print("loading tree label:", len(label_dic))
    if not os.path.exists(tmp_dict_save_path):
        # 从本地读取文本数据的id形式
        print('load sparse matrix...')
        text_ids_npz = load_sparse(npz_path)
        # 读取全部path对应的id
        print('load path ids...')
        with open(path_ids_path, 'r') as f:
            path_ids = [line.strip().split('\t')[0] for line in f.readlines()]
        # 建立id-path字典，并初始化
        print('construct id-paths dict')
        id2paths_dict = {}
        for i in label_dic.keys():
            id2paths_dict[i] = []
        path_num = len(path_ids)
        for path_cnt, i in enumerate(range(path_num)):
            if (path_cnt + 1) % 1000 == 0:
                print('dealing %d/%d' % (path_cnt + 1, path_num))
            tmp_id = text_ids_npz[i].nonzero()[1].tolist()
            id2paths_dict[path_ids[i]].append(tmp_id)

        # 将句子表示编码成相同的长度
        print('pad...')
        for id in id2paths_dict.keys():
            id2paths_dict[id] = pad_zero(id2paths_dict[id])
        # 将数据保存到本地，避免每次都处理
        print('save dict to', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'w')as f:
            json.dump(id2paths_dict, f)
    else:
        print('dict already exists,load from', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'r')as f:
            id2paths_dict = json.load(f)
    # 加载训练集id并转换成模型的输入x,y
    l1, l2, l3, l4 = 0, 0, 0, 0
    x_train, y_train = [], []
    per_class_threshold = int(_args.train_threshold / _args.class_num)
    with open(train_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        # random.shuffle(train_ids)
        for id in train_ids:
            y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
            if l1 > per_class_threshold and y == [1, 0, 0, 0]:
                l1 -= 1
                continue
            if l2 > per_class_threshold and y == [0, 1, 0, 0]:
                l2 -= 1
                continue
            if l3 > per_class_threshold and y == [0, 0, 1, 0]:
                l3 -= 1
                continue
            if l4 > per_class_threshold and y == [0, 0, 0, 1]:
                l4 -= 1
                continue
            y_train.append(y)
            x_train.append(id2paths_dict[id])
    print("loading train set:", l1, l2, l3, l4)

    # 加载测试集id并转换成模型的输入x,y
    l1, l2, l3, l4 = 0, 0, 0, 0
    x_test, y_test = [], []
    with open(test_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        for id in train_ids:
            y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
            y_test.append(y)
            x_test.append(id2paths_dict[id])
    print("loading test set:", l1, l2, l3, l4)
    return x_train, y_train, x_test, y_test

    # 从本地加载路径数据,使用bert分词，而不是npz文件
    # 参数：训练集路径，测试集路径，保存中间结果字典的路径
    # def load_path_data_bert(train_data_path, test_data_path, label_path, tmp_dict_save_path):
    #     # 获取标签字典
    #     label_dic = get_label_dict(label_path)
    #     print("loading tree label:", len(label_dic))
    #     #
    #     bert_tokenizer = BertTokenizer.from_pretrained(bert_weight_type,
    #                                                    cache_dir=bert_weight_dir_path,
    #                                                    do_lower_case=True)
    #     if not os.path.exists(tmp_dict_save_path):
    #         # 读取全部path对应的id
    #         print('load path ids...')
    #         with open(path_ids_path, 'r')as f:
    #             path_ids = [line.strip().split('\t')[0] for line in f.readlines()]
    #         with open(path_texts_path, 'r')as f:
    #             path_texts = [line.strip() for line in f.readlines()]
    #         # 建立id-path字典，并初始化
    #         print('construct id-paths dict')
    #         id2paths_dict = {}
    #         for i in label_dic.keys():
    #             id2paths_dict[i] = []
    #         path_num = len(path_ids)
    #         for path_cnt, i in enumerate(range(path_num)):
    #             if (path_cnt + 1) % 1000 == 0:
    #                 print('dealing %d/%d' % (path_cnt + 1, path_num))
    #             text = path_texts[i]
    #             tmp_id = bert_tokenizer.encode(text)
    #             id2paths_dict[path_ids[i]].append(tmp_id)
    #
    #         # 将句子表示编码成相同的长度
    #         print('pad...')
    #         for id in id2paths_dict.keys():
    #             id2paths_dict[id] = pad_zero(id2paths_dict[id], 128)
    #         # 将数据保存到本地，避免每次都处理
    #         print('save dict to', tmp_dict_save_path)
    #         with open(tmp_dict_save_path, 'w')as f:
    #             json.dump(id2paths_dict, f)
    #     else:
    #         print('dict already exists,load from', tmp_dict_save_path)
    #         with open(tmp_dict_save_path, 'r')as f:
    #             id2paths_dict = json.load(f)
    #     # 加载训练集id并转换成模型的输入x,y
    #     l1, l2, l3, l4 = 0, 0, 0, 0
    #     x_train, y_train = [], []
    #     per_class_threshold = int(_args.train_threshold / _args.class_num)
    #     with open(train_data_path, 'r')as f:
    #         train_ids = [line.strip() for line in f.readlines()]
    #         # random.shuffle(train_ids)
    #         for id in train_ids:
    #             y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
    #             if l1 > per_class_threshold and y == [1, 0, 0, 0]:
    #                 l1 -= 1
    #                 continue
    #             if l2 > per_class_threshold and y == [0, 1, 0, 0]:
    #                 l2 -= 1
    #                 continue
    #             if l3 > per_class_threshold and y == [0, 0, 1, 0]:
    #                 l3 -= 1
    #                 continue
    #             if l4 > per_class_threshold and y == [0, 0, 0, 1]:
    #                 l4 -= 1
    #                 continue
    #             y_train.append(y)
    #             x_train.append(id2paths_dict[id])
    #     print("loading train set:", l1, l2, l3, l4)
    # # 加载测试集id并转换成模型的输入x,y
    # l1, l2, l3, l4 = 0, 0, 0, 0
    # x_test, y_test = [], []
    # with open(test_data_path, 'r')as f:
    #     train_ids = [line.strip() for line in f.readlines()]
    #     for id in train_ids:
    #         y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
    #         y_test.append(y)
    #         x_test.append(id2paths_dict[id])
    # print("loading test set:", l1, l2, l3, l4)
    # return x_train, y_train, x_test, y_test


# 从本地加载路径数据
# 参数：训练集路径，测试集路径，保存中间结果字典的路径
def load_path_data_pheme(train_data_path, test_data_path, label_path, tmp_dict_save_path, npz_path):
    # 获取标签字典
    label_dic = get_label_dict(label_path)
    print("loading tree label:", len(label_dic))
    if not os.path.exists(tmp_dict_save_path):
        # 从本地读取文本数据的id形式
        print('load sparse matrix...')
        text_ids_npz = load_sparse(npz_path)
        # 读取全部path对应的id
        print('load path ids...')
        with open(path_ids_path, 'r') as f:
            path_ids = [line.strip().split('\t')[0] for line in f.readlines()]
        # 建立id-path字典，并初始化
        print('construct id-paths dict')
        id2paths_dict = {}
        for i in label_dic.keys():
            id2paths_dict[i] = []
        path_num = len(path_ids)
        for path_cnt, i in enumerate(range(path_num)):
            if (path_cnt + 1) % 1000 == 0:
                print('dealing %d/%d' % (path_cnt + 1, path_num))
            tmp_id = text_ids_npz[i].nonzero()[1].tolist()
            id2paths_dict[path_ids[i]].append(tmp_id)

        # 将句子表示编码成相同的长度
        print('pad...')
        for id in id2paths_dict.keys():
            id2paths_dict[id] = pad_zero(id2paths_dict[id])
        # 将数据保存到本地，避免每次都处理
        print('save dict to', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'w')as f:
            json.dump(id2paths_dict, f)
    else:
        print('dict already exists,load from', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'r')as f:
            id2paths_dict = json.load(f)
    # 加载训练集id并转换成模型的输入x,y
    l1, l2, l3 = 0, 0, 0
    x_train, y_train = [], []
    per_class_threshold = int(_args.train_threshold / _args.class_num)
    with open(train_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        # random.shuffle(train_ids)
        for id in train_ids:
            y, l1, l2, l3 = loadLabelPheme(label_dic[id], l1, l2, l3)
            if l1 > per_class_threshold and y == [1, 0, 0]:
                l1 -= 1
                continue
            if l2 > per_class_threshold and y == [0, 1, 0]:
                l2 -= 1
                continue
            if l3 > per_class_threshold and y == [0, 0, 1]:
                l3 -= 1
                continue
            y_train.append(y)
            x_train.append(id2paths_dict[id])
    print("loading train set:", l1, l2, l3)

    # 加载测试集id并转换成模型的输入x,y
    l1, l2, l3 = 0, 0, 0
    x_test, y_test = [], []
    with open(test_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        for id in train_ids:
            y, l1, l2, l3 = loadLabelPheme(label_dic[id], l1, l2, l3)
            y_test.append(y)
            x_test.append(id2paths_dict[id])
    print("loading test set:", l1, l2, l3)
    return x_train, y_train, x_test, y_test


# 从本地加载路径数据
# 参数：训练集路径，测试集路径，保存中间结果字典的路径
def load_path_data_weibo(train_data_path, test_data_path, label_path, tmp_dict_save_path, npz_path):
    # 获取标签字典
    label_dic = get_label_dict(label_path)
    print("loading tree label:", len(label_dic))
    if not os.path.exists(tmp_dict_save_path):
        # 从本地读取文本数据的id形式
        print('load sparse matrix...')
        text_ids_npz = load_sparse(npz_path)
        # 读取全部path对应的id
        print('load path ids...')
        with open(path_ids_path, 'r') as f:
            path_ids = [line.strip().split('\t')[0] for line in f.readlines()]
        # 建立id-path字典，并初始化
        print('construct id-paths dict')
        id2paths_dict = {}
        for i in label_dic.keys():
            id2paths_dict[i] = []
        path_num = len(path_ids)
        for path_cnt, i in enumerate(range(path_num)):
            if (path_cnt + 1) % 1000 == 0:
                print('dealing %d/%d' % (path_cnt + 1, path_num))
            tmp_id = text_ids_npz[i].nonzero()[1].tolist()
            id2paths_dict[path_ids[i]].append(tmp_id)

        # 将句子表示编码成相同的长度
        print('pad...')
        for id in id2paths_dict.keys():
            id2paths_dict[id] = pad_zero(id2paths_dict[id])
        # 将数据保存到本地，避免每次都处理
        print('save dict to', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'w')as f:
            json.dump(id2paths_dict, f)
    else:
        print('dict already exists,load from', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'r')as f:
            id2paths_dict = json.load(f)
    # 加载训练集id并转换成模型的输入x,y
    l1, l2 = 0, 0
    x_train, y_train = [], []
    per_class_threshold = int(_args.train_threshold / _args.class_num)
    with open(train_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        # random.shuffle(train_ids)
        for id in train_ids:
            y, l1, l2 = loadLabelWeibo(label_dic[id], l1, l2)
            if l1 > per_class_threshold and y == [1, 0]:
                l1 -= 1
                continue
            if l2 > per_class_threshold and y == [0, 1]:
                l2 -= 1
                continue
            y_train.append(y)
            x_train.append(id2paths_dict[id])
    print("loading train set:", l1, l2)

    # 加载测试集id并转换成模型的输入x,y
    l1, l2 = 0, 0
    x_test, y_test = [], []
    with open(test_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        for id in train_ids:
            y, l1, l2 = loadLabelWeibo(label_dic[id], l1, l2)
            y_test.append(y)
            x_test.append(id2paths_dict[id])
    print("loading test set:", l1, l2)
    return x_train, y_train, x_test, y_test


# 返回id-index词典，根据每个id的text获取对应text在（N,5001）矩阵中的index
def get_index_dict(id_text_path):
    with open(id_text_path, 'r', encoding='utf-8')as f:
        ids = [line.strip().split('\t')[0] for line in f.readlines()]
    index_dict = {}
    for cnt, i in enumerate(ids):
        index_dict[ids[cnt]] = cnt
    return index_dict


# 从本地加载路径数据
# 参数：训练集路径，测试集路径，保存中间结果字典的路径
def load_path_data_for_early_detection(train_data_path, test_data_path, label_path,
                                       tmp_dict_save_path, path_node_ids_dict_path,
                                       node_id2token_id_dict, if_response,
                                       if_random_shuffle):
    # 获取标签字典
    label_dic = get_label_dict(label_path)
    print("loading tree label:", len(label_dic))
    if not os.path.exists(tmp_dict_save_path) or True:
        # 从本地读取路径的node id表示
        path_node_ids = load_dict_json(path_node_ids_dict_path)
        # 从本地加载node id->token id的字典
        node2token_ids = load_dict_json(node_id2token_id_dict)
        # 建立id-path字典，并初始化
        print('construct id-paths dict')
        id2paths_dict = {}
        for path_cnt, key in enumerate(path_node_ids.keys()):
            if (path_cnt + 1) % 100 == 0:
                print('dealing %d/%d' % (path_cnt + 1, len(path_node_ids)))
            token_ids = []
            id2paths_dict[key] = []
            # 如果是全部路径的编码，且当前路径为空，则添加根结点作为唯一路径
            if len(path_node_ids[key]) == 0:
                path_node_ids[key] = [[key, ]]
            for path_nodes in path_node_ids[key]:
                tmp = []
                if if_response:
                    for id in path_nodes[1:]:
                        tmp += node2token_ids[id]
                else:
                    for id in path_nodes:
                        tmp += node2token_ids[id]
                tmp = sorted(tmp)
                token_ids.append(tmp)
            if if_random_shuffle:
                token_ids = shuffle_path_token_ids(token_ids)
            id2paths_dict[key] = token_ids

        # 将句子表示编码成相同的长度
        print('pad...')
        for id in id2paths_dict.keys():
            id2paths_dict[id] = pad_zero(id2paths_dict[id])
        # 将数据保存到本地，避免每次都处理
        print('save dict to', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'w')as f:
            json.dump(id2paths_dict, f)
    else:
        print('dict already exists,load from', tmp_dict_save_path)
        with open(tmp_dict_save_path, 'r')as f:
            id2paths_dict = json.load(f)
    # 加载训练集id并转换成模型的输入x,y
    l1, l2, l3, l4 = 0, 0, 0, 0
    x_train, y_train = [], []
    per_class_threshold = int(_args.train_threshold / _args.class_num)
    with open(train_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        # random.shuffle(train_ids)
        for id in train_ids:
            y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
            if l1 > per_class_threshold and y == [1, 0, 0, 0]:
                l1 -= 1
                continue
            if l2 > per_class_threshold and y == [0, 1, 0, 0]:
                l2 -= 1
                continue
            if l3 > per_class_threshold and y == [0, 0, 1, 0]:
                l3 -= 1
                continue
            if l4 > per_class_threshold and y == [0, 0, 0, 1]:
                l4 -= 1
                continue
            y_train.append(y)
            x_train.append(id2paths_dict[id])
    print("loading train set:", l1, l2, l3, l4)

    # 加载测试集id并转换成模型的输入x,y
    l1, l2, l3, l4 = 0, 0, 0, 0
    x_test, y_test = [], []
    with open(test_data_path, 'r')as f:
        train_ids = [line.strip() for line in f.readlines()]
        for id in train_ids:
            y, l1, l2, l3, l4 = loadLabel(label_dic[id], l1, l2, l3, l4)
            y_test.append(y)
            x_test.append(id2paths_dict[id])
    print("loading test set:", l1, l2, l3, l4)
    return x_train, y_train, x_test, y_test


# 将不同路径中的单词打乱，同时保证路径数和每条路径的长度都和原来相同
def shuffle_path_token_ids(path_token_ids):
    all_tokens = []
    for path_ids in path_token_ids:
        all_tokens += path_ids
    random.shuffle(all_tokens)
    new_token_ids = []
    index = 0
    for path_ids in path_token_ids:
        tmp = all_tokens[index:index + len(path_ids)]
        new_token_ids.append(sorted(tmp))
        index += len(path_ids)
    return new_token_ids


def load_lst(path):
    with open(path, 'r')as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def save_lst(path, lines):
    with open(path, 'w')as f:
        f.write('\n'.join(lines) + '\n')
