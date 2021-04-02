# 一些基础常见的代码
from config import *
import random
import numpy as np


# 为随机数产生函数设置固定种子，便于实验复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 根据max_len为列表补齐，如果没有指定max_len，则根据数据自动计算
def pad_zero(lst, max_len=-1):
    if max_len == -1:
        for i in lst:
            max_len = max(max_len, len(i))
    for cnt in range(len(lst)):
        lst[cnt] = lst[cnt] + [0] * (max_len - len(lst[cnt]))
        # 截断过长的部分；当max-len是手动设置的参数时，需要截断长度超过的部分
        lst[cnt] = lst[cnt][:max_len]
    return lst
