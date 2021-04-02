# 为wae无监督训练预处理文本数据
import argparse
import json
import os
from typing import List
import numpy as np
import spacy
from allennlp.data.tokenizers import Tokenizer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import random
from data_io import save_sparse, write_to_json


def load_data(limit, data_path: str, tokenize: bool = True, tokenizer_type: str = "just_spaces") -> List[str]:
    if tokenizer_type == "just_spaces":
        # 有待修改
        tokenizer = Tokenizer(5000)
        # tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en_core_web_sm')
        tokenizer = Tokenizer(nlp.vocab)
    tokenized_examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                example = json.loads(line)
            else:
                example = {"text": line.strip()}
            if tokenize:
                if tokenizer_type == 'just_spaces':
                    tokens = list(map(str, tokenizer.split_words(example['text'])))
                elif tokenizer_type == 'spacy':
                    tokens = list(map(str, tokenizer(example['text'])))
                text = ' '.join(tokens)
            else:
                text = example['text']
            tokenized_examples.append(text)
            if len(tokenized_examples) >= limit:
                break
    return tokenized_examples


def pre4wae(serialization_dir, train_path, vocab_size, ):
    # 文件夹存在判断与创建
    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    # 读取全部回复文本（1000000是上限；如果文件中超过上限，则后面的不处理，用于处理数据过多的文件，100000手动设置）
    raw_tokenized_all_examples = load_data(100000, train_path, True, "spacy")
    examples_num = len(raw_tokenized_all_examples)
    # 按照比例划分作为训练和验证集
    train_num = int(examples_num * 0.9)
    # 复制一遍；便于打乱顺序等操作
    random_all_examples = [line for line in raw_tokenized_all_examples]
    random.shuffle(random_all_examples)

    # 划分shuffle后的数据；打乱是为了保证训练集基本覆盖全部树的路径
    tokenized_train_examples = random_all_examples[:train_num]
    tokenized_dev_examples = random_all_examples[train_num:]
    print("fitting count vectorizer...")

    count_vectorizer = CountVectorizer(stop_words='english',
                                       max_features=vocab_size,
                                       token_pattern=r'\b[^\d\W]{2,20}\b'
                                       )
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english',
    #                                    max_features=args.vocab_size,
    #                                    token_pattern=r'\b[^\d\W]{3,30}\b')

    count_vectorizer.fit(tqdm(random_all_examples))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))
    vectorized_raw_all_examples = count_vectorizer.transform(tqdm(raw_tokenized_all_examples))
    reference_vectorizer = CountVectorizer(stop_words='english',
                                           token_pattern=r'\b[^\d\W]{2,20}\b')

    print("fitting reference corpus using development data...")
    reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))

    reference_vocabulary = reference_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack(
        (np.array([0] * len(tokenized_train_examples))[:, None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack(
        (np.array([0] * len(tokenized_dev_examples))[:, None], vectorized_dev_examples))
    vectorized_raw_all_examples = sparse.hstack(
        (np.array([0] * len(raw_tokenized_all_examples))[:, None], vectorized_raw_all_examples))

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(),
                      (np.array(vectorized_raw_all_examples.sum(0)) / vocab_size).squeeze()))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(serialization_dir, "dev.npz"))
    save_sparse(vectorized_raw_all_examples, os.path.join(serialization_dir, "all.npz"))
    if not os.path.isdir(os.path.join(serialization_dir, "reference")):
        os.mkdir(os.path.join(serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(serialization_dir, "reference", "ref.vocab.json"))
    write_to_json(bgfreq, os.path.join(serialization_dir, "path.bgfreq"))

    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(),
                       os.path.join(vocabulary_dir, "vocab.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))


def pre4wae_chinese(serialization_dir, train_path, vocab_size):
    # 文件夹存在判断与创建
    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    # 读取全部回复文本（1000000是上限；如果文件中超过上限，则后面的不处理，用于处理数据过多的文件，100000手动设置）
    raw_tokenized_all_examples = load_data(1000000, train_path, True, "spacy")
    examples_num = len(raw_tokenized_all_examples)
    # 按照比例划分作为训练和验证集
    train_num = int(examples_num * 0.9)
    # 复制一遍；便于打乱顺序等操作
    random_all_examples = [line for line in raw_tokenized_all_examples]
    random.shuffle(random_all_examples)

    # 划分shuffle后的数据；打乱是为了保证训练集基本覆盖全部树的路径
    tokenized_train_examples = random_all_examples[:train_num]
    tokenized_dev_examples = random_all_examples[train_num:]
    print("fitting count vectorizer...")

    count_vectorizer = CountVectorizer(stop_words='english',
                                       max_features=vocab_size,
                                       # token_pattern=r'\b[^\d\W]{2,20}\b'
                                       )
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english',
    #                                    max_features=args.vocab_size,
    #                                    token_pattern=r'\b[^\d\W]{3,30}\b')

    count_vectorizer.fit(tqdm(random_all_examples))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))
    vectorized_raw_all_examples = count_vectorizer.transform(tqdm(raw_tokenized_all_examples))
    reference_vectorizer = CountVectorizer(stop_words='english',
                                           token_pattern=r'\b[^\d\W]{2,20}\b')

    print("fitting reference corpus using development data...")
    reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))

    reference_vocabulary = reference_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack(
        (np.array([0] * len(tokenized_train_examples))[:, None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack(
        (np.array([0] * len(tokenized_dev_examples))[:, None], vectorized_dev_examples))
    vectorized_raw_all_examples = sparse.hstack(
        (np.array([0] * len(raw_tokenized_all_examples))[:, None], vectorized_raw_all_examples))

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(),
                      (np.array(vectorized_raw_all_examples.sum(0)) / vocab_size).squeeze()))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(serialization_dir, "dev.npz"))
    save_sparse(vectorized_raw_all_examples, os.path.join(serialization_dir, "all.npz"))
    if not os.path.isdir(os.path.join(serialization_dir, "reference")):
        os.mkdir(os.path.join(serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(serialization_dir, "reference", "ref.vocab.json"))
    write_to_json(bgfreq, os.path.join(serialization_dir, "path.bgfreq"))

    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(),
                       os.path.join(vocabulary_dir, "vocab.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))


def pre4wae_response(serialization_dir, train15_path, train16_path, vocab_size):
    # 文件夹存在判断与创建
    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    # 读取全部回复文本（100000是上限；如果文件中超过上限，则后面的不处理，用于处理数据过多的文件，100000手动设置）
    raw_tokenized15_all_examples = load_data(100000, train15_path, True, "spacy")
    raw_tokenized16_all_examples = load_data(100000, train16_path, True, "spacy")
    raw_tokenized_all_examples = raw_tokenized15_all_examples + raw_tokenized16_all_examples
    # 按照比例划分作为训练和验证集
    examples_num = len(raw_tokenized_all_examples)
    train_num = int(examples_num * 0.9)
    dev_num = examples_num - train_num
    path15_num = len(raw_tokenized15_all_examples)
    path16_num = len(raw_tokenized16_all_examples)

    # 复制一遍；便于打乱顺序等操作
    random_all_examples = [line for line in raw_tokenized_all_examples]
    random.shuffle(random_all_examples)

    # 划分shuffle后的数据；打乱是为了保证训练集基本覆盖全部树的路径
    tokenized_train_examples = random_all_examples[:train_num]
    tokenized_dev_examples = random_all_examples[train_num:]
    print("fitting count vectorizer...")

    count_vectorizer = CountVectorizer(stop_words='english',
                                       max_features=vocab_size,
                                       token_pattern=r'\b[^\d\W]{2,20}\b'
                                       )
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english',
    #                                    max_features=args.vocab_size,
    #                                    token_pattern=r'\b[^\d\W]{3,30}\b')

    count_vectorizer.fit(tqdm(random_all_examples))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))
    vectorized_raw15_examples = count_vectorizer.transform(tqdm(raw_tokenized15_all_examples))
    vectorized_raw16_examples = count_vectorizer.transform(tqdm(raw_tokenized16_all_examples))

    reference_vectorizer = CountVectorizer(stop_words='english',
                                           token_pattern=r'\b[^\d\W]{2,20}\b')
    print("fitting reference corpus using development data...")
    reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))
    reference_vocabulary = reference_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack(
        (np.array([0] * train_num)[:, None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack(
        (np.array([0] * dev_num)[:, None], vectorized_dev_examples))
    vectorized_raw15_examples = sparse.hstack(
        (np.array([0] * path15_num)[:, None], vectorized_raw15_examples))
    vectorized_raw16_examples = sparse.hstack(
        (np.array([0] * path16_num)[:, None], vectorized_raw16_examples))
    # generate background frequency
    # print("generating background frequency...")
    # bgfreq = dict(zip(count_vectorizer.get_feature_names(),
    #                   (np.array(vectorized_raw_all_examples.sum(0)) / vocab_size).squeeze()))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(serialization_dir, "dev.npz"))
    save_sparse(vectorized_raw15_examples, os.path.join(serialization_dir, "all15.npz"))
    save_sparse(vectorized_raw16_examples, os.path.join(serialization_dir, "all16.npz"))
    if not os.path.isdir(os.path.join(serialization_dir, "reference")):
        os.mkdir(os.path.join(serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(serialization_dir, "reference", "ref.vocab.json"))
    # write_to_json(bgfreq, os.path.join(serialization_dir, "path.bgfreq"))

    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(),
                       os.path.join(vocabulary_dir, "vocab.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))


def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')
