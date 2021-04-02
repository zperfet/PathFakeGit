from config import *
import random
from data_io import load_lst, save_lst


# 划分数据，获取训练集和测试集
# 输入：待划分的ids,保存的文件夹路径；几重交叉验证
# 输出：划分后的训练、测试集，每一行一个source id
def dataset_split(ids, fold_save_dir_path, k_fold):
    random.shuffle(ids)
    per_fold_num = len(ids) // k_fold
    for ki in range(k_fold):
        fold_train_path = join(fold_save_dir_path, 'TrainPheme5Fold%d.txt' % ki)
        fold_test_path = join(fold_save_dir_path, 'TestPheme5Fold%d.txt' % ki)
        test_ids = ids[ki * per_fold_num:min(len(ids), (ki + 1) * per_fold_num)]
        train_ids = list(set(ids) - set(test_ids))
        random.shuffle(train_ids)
        random.shuffle(test_ids)
        save_lst(fold_train_path, train_ids)
        save_lst(fold_test_path, test_ids)


pheme_label_path = join(data_dir_path, 'labelPheme5.txt')
lines = load_lst(pheme_label_path)
pheme_ids = [line.split('\t')[0] for line in lines]
dataset_split(pheme_ids, fold_dir_path, 5)
