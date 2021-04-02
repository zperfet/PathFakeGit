# 获取tfidf格式表达的PHEME数据集
# 作用：作为BiGCN模型的输入，测试混合Pheme5数据集在BiGCN上的性能
# 1: root-id -- an unique identifier describing the tree (tweetid of the root);
# 2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;
# 3: index-of-the-current-tweet -- an index number of the current tweet;
# 4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;
# 5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;
# 6: list-of-index-and-counts -- the rest of the line contains space separated index-count
# pairs, where a index-count pair is in format of "index:count", E.g.,
# "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)

from config import data_dir_path
from config import join, os
from data_io import load_lst, save_lst

pheme_tree_dir_path = join(data_dir_path, 'TreePheme5')
pheme_id2text_path = join(data_dir_path, 'pheme_all_text.txt')
pheme_tfidf_save_path = join(data_dir_path, 'pheme_tfidf.txt')


# 将文本转换为tfidf格式
def text2tfidf(text, token2id_dict):
    token_lst = text.split(' ')
    cnt_dict = {}
    tokens_set = set(token2id_dict.keys())
    for token in token_lst:
        if token not in tokens_set: continue
        tfidf = token2id_dict[token]
        if tfidf not in cnt_dict.keys():
            cnt_dict[tfidf] = 1
        else:
            cnt_dict[tfidf] = cnt_dict[tfidf] + 1
    tfidf_id_cnt_line = []
    for key in cnt_dict.keys():
        tfidf_id_cnt_line.append(str(key) + ':' + str(cnt_dict[key]))
    res = ' '.join(tfidf_id_cnt_line) or '0:1'
    return res


# 返回根据id获取文本的字典
def get_id2text_dict(id2_text_path):
    id2texts_lines = load_lst(id2_text_path)
    id2text = {}
    for l in id2texts_lines:
        l = l.split('\t')
        id2text[l[0]] = l[1]
    return id2text


def get_token2id_dict(vocab_path):
    tokens = load_lst(vocab_path)
    tokens = tokens[:5000]
    token2id_dict = {}
    for cnt, token in enumerate(tokens):
        token2id_dict[token] = cnt
    return token2id_dict


# 根据推特id获取文本的字典
id2text_dict = get_id2text_dict(pheme_id2text_path)
# 根据单词获取单词id
pheme_vocab_path = join(data_dir_path, 'pathPheme5vocab4random', 'vocabulary', 'vocab.txt')
token2id_dict = get_token2id_dict(pheme_vocab_path)

# 树文件的每一个文件名
names = os.listdir(pheme_tree_dir_path)
# 最后的结果
TFIDF_lines = []
for name_cnt, name in enumerate(names):
    if name_cnt % 100 == 0:
        print("处理:", name_cnt)
    pheme_json_path = join(pheme_tree_dir_path, name)
    source_id = name.split('.')[0]
    lines = load_lst(pheme_json_path)
    node_set = set()
    node_dict = {}
    # 父节点的集合
    father_set = set()
    # 最大句子长度
    max_token_sen = 0
    # 赋予每个id一个index
    for line in lines:
        line = line.split('\t')
        if len(line) == 1: continue
        father_set.add(line[0])
        if line[0] not in node_set:
            node_set.add(line[0])
            node_dict[line[0]] = str(len(node_set))
        if line[2] not in node_set:
            node_set.add(line[2])
            node_dict[line[2]] = str(len(node_set))
    # 将源文本转换为tfidf格式
    source_text = id2text_dict[source_id]
    source_tfidf = text2tfidf(source_text, token2id_dict)
    max_token_sen = source_tfidf.count(':')
    source_tfidf_line = [source_id, "None", "1", source_tfidf]
    tmp_TFIDF_lines = []
    tmp_TFIDF_lines.append(source_tfidf_line)
    for line in lines:
        tfidf_line = [source_id, ]
        line = line.split('\t')
        if len(line) == 1: continue
        tfidf_line.append(node_dict[line[0]])
        tfidf_line.append(node_dict[line[2]])
        node_text = id2text_dict[line[2]]
        node_tfidf = text2tfidf(node_text, token2id_dict)
        max_token_sen = max(max_token_sen, node_tfidf.count(':'))
        tfidf_line.append(node_tfidf)
        tmp_TFIDF_lines.append(tfidf_line)
    tmp_TFIDF_lines = [line[:3] + [str(len(father_set)), str(max_token_sen)] + line[3:] for line in tmp_TFIDF_lines]
    tmp_TFIDF_lines = ['\t'.join(line) for line in tmp_TFIDF_lines]
    TFIDF_lines += tmp_TFIDF_lines
save_lst(pheme_tfidf_save_path, TFIDF_lines)
