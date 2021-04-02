import argparse


# todo合并args函数中公共的设置，避免混乱
def get_response_twitter_args():
    parser = argparse.ArgumentParser()
    # 实验设置
    parser.add_argument('--cuda', type=int, default=1, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
    parser.add_argument('--seed', type=int, default=100, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
    parser.add_argument('--novae', type=int, default=0, help='选择random+vae还是random only')
    parser.add_argument('--norandom', type=int, default=1, help='选择random+vae还是random only')
    parser.add_argument('--epoch', type=int, default=150, help='epoch数')
    parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
    parser.add_argument('--lr', type=float, default=0.05, help='学习率')
    parser.add_argument('--train_threshold', type=int, default=2000,
                        help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
    parser.add_argument('--fold', type=int, default=2, help='使用K（5）重交叉验证中的哪一个')
    parser.add_argument('--dataset', type=str, default="Twitter15", help='使用哪个数据集；Twitter15|Twitter16')
    parser.add_argument('--freeze_wae', type=int, default=0, help='是否冻结wae的encoder权重训练')
    parser.add_argument('--wae_dropout', type=float, default=0.5, help='wae输出的dropout')
    parser.add_argument('--encoder_index', type=str, default='80_0.128', help='使用哪一个预训练好的encoder权重')
    # time_interval = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 120,100000000]
    parser.add_argument('--interval', type=int, default=120, help='early detection的最大时间,')
    parser.add_argument('--path_shuffle', type=bool, default=False, help='是否shuffle路径文本')

    # 模型设置
    parser.add_argument('--class_num', type=int, default=4, help='数据集类别数')
    parser.add_argument('--random_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    parser.add_argument('--response_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    # actually，range(30,70) will all be fine,
    parser.add_argument('--vae_dim', type=int, default=50, help='vae dim')
    parser.add_argument('--random_dim', type=int, default=50, help='random dim')
    parser.add_argument('--bert_dim', type=int, default=768, help='random dim')
    parser.add_argument('--verbose', type=int, default=1, help='whether print log information')
    parser.add_argument('--load_wae_encoder', type=int, default=1, help='是否加载wae预训练好的权重')
    parser.add_argument('--rate_lambda', type=float, default=0.6, help='wae和ppa的占比设置')
    args = parser.parse_args()
    return args


def get_response_weibo_args():
    parser = argparse.ArgumentParser()
    # 实验设置
    parser.add_argument('--cuda', type=int, default=0, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
    parser.add_argument('--seed', type=int, default=100, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
    parser.add_argument('--novae', type=int, default=0, help='选择random+vae还是random only')
    parser.add_argument('--norandom', type=int, default=0, help='选择random+vae还是random only')
    parser.add_argument('--epoch', type=int, default=25, help='epoch数')
    parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
    parser.add_argument('--lr', type=float, default=0.05, help='学习率')
    parser.add_argument('--train_threshold', type=int, default=1000,
                        help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
    parser.add_argument('--fold', type=int, default=0, help='使用K（5）重交叉验证中的哪一个')
    parser.add_argument('--dataset', type=str, default="Weibo", help='使用哪个数据集；Twitter15|Twitter16')
    parser.add_argument('--freeze_wae', type=int, default=0, help='是否冻结wae的encoder权重训练')
    parser.add_argument('--wae_dropout', type=float, default=0.5, help='wae输出的dropout')
    parser.add_argument('--encoder_index', type=str, default='80_0.128', help='使用哪一个预训练好的encoder权重')
    # time_interval = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 120,100000000]
    parser.add_argument('--interval', type=int, default=100000000, help='early detection的最大时间,')
    parser.add_argument('--path_shuffle', type=bool, default=False, help='是否shuffle路径文本')

    # 模型设置
    parser.add_argument('--class_num', type=int, default=2, help='数据集类别数')
    parser.add_argument('--random_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    parser.add_argument('--response_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    # actually，range(30,70) will all be fine,
    parser.add_argument('--vae_dim', type=int, default=50, help='vae dim')
    parser.add_argument('--random_dim', type=int, default=50, help='random dim')
    parser.add_argument('--bert_dim', type=int, default=768, help='random dim')
    parser.add_argument('--verbose', type=int, default=1, help='whether print log information')
    parser.add_argument('--load_wae_encoder', type=int, default=1, help='是否加载wae预训练好的权重')
    parser.add_argument('--rate_lambda', type=float, default=0.6, help='wae和ppa的占比设置')
    args = parser.parse_args()
    return args


def get_response_pheme5_args():
    parser = argparse.ArgumentParser()
    # 实验设置
    parser.add_argument('--cuda', type=int, default=1, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
    parser.add_argument('--seed', type=int, default=100, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
    parser.add_argument('--novae', type=int, default=1, help='选择random+vae还是random only')
    parser.add_argument('--norandom', type=int, default=0, help='选择random+vae还是random only')
    parser.add_argument('--epoch', type=int, default=150, help='epoch数')
    parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
    parser.add_argument('--lr', type=float, default=0.02, help='学习率')
    parser.add_argument('--train_threshold', type=int, default=2000,
                        help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
    # fold4 f1 0.355;fold3 0.374;fold2 0.153;fold1 0.290;fold0 0.350；均值：0.304
    # mix fold0 0.93 fold1 0.88
    parser.add_argument('--fold', type=int, default=0, help='使用K（5）重交叉验证中的哪一个')
    parser.add_argument('--dataset', type=str, default="Pheme5", help='使用哪个数据集；Twitter15|Twitter16')
    parser.add_argument('--freeze_wae', type=int, default=0, help='是否冻结wae的encoder权重训练')
    parser.add_argument('--wae_dropout', type=float, default=0.5, help='wae输出的dropout')
    parser.add_argument('--encoder_index', type=str, default='80_0.128', help='使用哪一个预训练好的encoder权重')
    # time_interval = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 120,100000000]
    parser.add_argument('--interval', type=int, default=100000000, help='early detection的最大时间,')
    parser.add_argument('--path_shuffle', type=bool, default=False, help='是否shuffle路径文本')

    # 模型设置
    parser.add_argument('--class_num', type=int, default=3, help='数据集类别数')
    parser.add_argument('--random_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    parser.add_argument('--response_vocab_dim', type=int, default=5000 + 1, help='词汇数（第一个是UNKNOWN）')
    # actually，range(30,70) will all be fine,
    parser.add_argument('--vae_dim', type=int, default=50, help='vae dim')
    parser.add_argument('--random_dim', type=int, default=50, help='random dim')
    parser.add_argument('--bert_dim', type=int, default=768, help='random dim')
    parser.add_argument('--verbose', type=int, default=1, help='whether print log information')
    parser.add_argument('--load_wae_encoder', type=int, default=1, help='是否加载wae预训练好的权重')
    parser.add_argument('--rate_lambda', type=float, default=0.6, help='wae和ppa的占比设置')
    args = parser.parse_args()
    return args


# def get_vae_path_args():
#     parser = argparse.ArgumentParser()
#     # 实验设置
#     parser.add_argument('--cuda', type=int, default=1, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
#     parser.add_argument('--seed', type=int, default=100, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
#     parser.add_argument('--novae', type=int, default=1, help='选择random+vae还是random only')
#     parser.add_argument('--norandom', type=int, default=0, help='选择random+vae还是random only')
#     parser.add_argument('--epoch', type=int, default=100, help='epoch数')
#     parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
#     parser.add_argument('--lr', type=float, default=0.05, help='学习率')
#     parser.add_argument('--train_threshold', type=int, default=200,
#                         help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
#     parser.add_argument('--fold', type=int, default=2, help='使用K（5）重交叉验证中的哪一个')
#     parser.add_argument('--dataset', type=str, default='Twitter15', help='使用哪个数据集；15|16')
#     parser.add_argument('--vam_name', type=str, default='path15_response20000',
#                         help='使用哪个预训练的vampire文件夹名称；默认路径会自动补齐')
#     parser.add_argument('--freeze_wae', type=int, default=0, help='是否冻结wae的encoder权重训练  ')
#     parser.add_argument('--wae_dropout', type=float, default=0.5, help='wae输出的dropout')
#     parser.add_argument('--encoder_index', type=str, default='43_0.087', help='使用哪一个预训练好的encoder权重')
#
#     # 模型设置
#     parser.add_argument('--class_num', type=int, default=4, help='数据集类别数')
#     parser.add_argument('--vocab_dim', type=int, default=20000 + 1, help='词汇数（第一个是UNKNOWN）')
#     parser.add_argument('--vae_dim', type=int, default=50, help='vae dim')
#     parser.add_argument('--random_dim', type=int, default=50, help='random dim')
#     parser.add_argument('--verbose', type=int, default=1, help='whether print log information')
#     parser.add_argument('--load_wae_encoder', type=int, default=1, help='是否加载wae预训练好的权重')
#     args = parser.parse_args()
#     return args
#
#
# def get_ma_args():
#     parser = argparse.ArgumentParser()
#     # 实验设置
#     parser.add_argument('--cuda', type=int, default=3, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
#     parser.add_argument('--seed', type=int, default=110, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
#     parser.add_argument('--epoch', type=int, default=100, help='epoch数')
#     parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
#     parser.add_argument('--lr', type=float, default=0.05, help='学习率')
#     parser.add_argument('--train_threshold', type=int, default=1200,
#                         help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
#     parser.add_argument('--fold', type=int, default=2, help='使用K（5）重交叉验证中的哪一个')
#     parser.add_argument('--dataset', type=str, default='Twitter15', help='使用哪个数据集；15|16')
#
#     # 模型设置
#     parser.add_argument('--class_num', type=int, default=4, help='数据集类别数')
#     parser.add_argument('--vocab_dim', type=int, default=5000, help='词汇数')
#     parser.add_argument('--random_dim', type=int, default=100, help='random dim')
#
#     args = parser.parse_args()
#     return args
#
#
# def get_ma_vae_args():
#     parser = argparse.ArgumentParser()
#     # 实验设置
#     parser.add_argument('--cuda', type=int, default=3, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
#     parser.add_argument('--seed', type=int, default=110, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
#     parser.add_argument('--epoch', type=int, default=100, help='epoch数')
#     parser.add_argument('--optim', type=str, default='adagrad', help='学习策略adagrad|adam')
#     parser.add_argument('--lr', type=float, default=0.05, help='学习率')
#     parser.add_argument('--train_threshold', type=int, default=1200,
#                         help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
#     parser.add_argument('--fold', type=int, default=2, help='使用K（5）重交叉验证中的哪一个')
#     parser.add_argument('--dataset', type=str, default='Twitter15', help='使用哪个数据集；15|16')
#     parser.add_argument('--vam_name', type=str, default='path15_random_debug',
#                         help='使用哪个预训练的vampire文件夹名称；默认路径会自动补齐')
#
#     # 模型设置
#     parser.add_argument('--class_num', type=int, default=4, help='数据集类别数')
#     parser.add_argument('--vocab_dim', type=int, default=5000, help='词汇数')
#     parser.add_argument('--random_dim', type=int, default=100, help='random dim')
#
#     args = parser.parse_args()
#     return args
#
#
# def get_ag_args():
#     parser = argparse.ArgumentParser()
#     # 实验设置
#     parser.add_argument('--cuda', type=int, default=1, help='设置使用cpu（-1）还是gpu|cuda（0,1,2,3）')
#     parser.add_argument('--seed', type=int, default=100, help="设置随机数种子；设置相同的种子可保证得到相同的结果")
#     parser.add_argument('--novae', type=int, default=0, help='选择random+vae还是random only')
#     parser.add_argument('--epoch', type=int, default=100, help='epoch数')
#     parser.add_argument('--optim', type=str, default='adam', help='学习策略adagrad|adam')
#     parser.add_argument('--lr', type=float, default=0.001, help='学习率')
#     parser.add_argument('--train_threshold', type=int, default=200,
#                         help='有监督训练使用的训练集样本数；用于测试vae在小样本数据集上的性能')
#     parser.add_argument('--fold', type=int, default=2, help='使用K（5）重交叉验证中的哪一个')
#     parser.add_argument('--dataset', type=str, default='Twitter15', help='使用哪个数据集；15|16')
#     parser.add_argument('--vam_name', type=str, default='ag_vam',
#                         help='使用哪个预训练的vampire文件夹名称；默认路径会自动补齐')
#
#     # 模型设置
#     parser.add_argument('--class_num', type=int, default=4, help='数据集类别数')
#     parser.add_argument('--vocab_dim', type=int, default=30000 + 1, help='词汇数（第一个是UNKNOWN）')
#     parser.add_argument('--vae_dim', type=int, default=81, help='vae dim')
#     parser.add_argument('--random_dim', type=int, default=50, help='random dim')
#
#     args = parser.parse_args()
#     return args


def print_args(args):
    args_dict = args.__dict__
    print('模型参数设置：')
    for k, v in args_dict.items():
        print(k, ':', v)


# 综合vae和path voting算法的参数，适合Twitter15|Twitter16
# 输入测试哪个数据集性能，T：Twitter；W：Weibo；P：Pheme5
cur_dataset = "W"
if cur_dataset is "T":
    _args = get_response_twitter_args()
elif cur_dataset is "W":
    _args = get_response_weibo_args()
elif cur_dataset is "P":
    _args = get_response_pheme5_args()
else:
    print("数据集设置错误")

# 马金TD_RvNN_Ma参数
# _ma_args = get_ma_args()

# 基于新数据集和VAE的Ma2019参数
# _ma_vae_args = get_ma_vae_args()

# 测试ag数据集的效果
# _ag_args = get_ag_args()
