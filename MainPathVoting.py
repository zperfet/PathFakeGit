from models.PathBased import PathVoting, PathWAE
from torch import optim
import datetime
from evaluate import *
from get_args import _args, print_args
from data_io import *
import sys
from torch.optim.lr_scheduler import StepLR


def main():
    print_args(_args)
    # 固定随机数种子
    setup_seed(_args.seed)
    print("Step1:processing data")

    x_train, y_train, x_test, y_test = load_path_data(train_path, test_path, vampire_id2paths_dict_path,
                                                      vampire_npz_path)

    print('Step2:build model')
    # model = PathVoting(_args.vocab_dim, _args.random_dim, _args.vae_dim, _args.class_num)
    model = PathWAE(_args.vocab_dim, wae_best_encoder_path, _args.random_dim, _args.vae_dim, _args.class_num)
    model.to(device)

    # 3. looping SGD
    print('Step3:start training')
    if _args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=_args.lr)
    elif _args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=_args.lr)
    else:
        print('optim %s not correct' % _args.optim)
        return
    # scheduler = StepLR(optimizer, 10, 0.5, -1)
    losses_5, losses = [], []
    num_examples_seen = 0
    indexs = list(range(len(y_train)))
    highest_acc = 0
    for epoch in range(1, _args.epoch + 1):
        # print('epoch：', epoch, '学习率：', scheduler.get_lr())
        # 每次训练打乱顺序
        random.shuffle(indexs)
        # 训练模型
        for cnt, i in enumerate(indexs):
            pred_y, loss = model.forward(torch.Tensor(x_train[i]).cuda(device).long(),
                                         torch.Tensor(y_train[i]).cuda(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().tolist())
            num_examples_seen += 1
            if (cnt + 1) % 500 == 0:
                break
        # scheduler.step()
        # cal loss & evaluate
        if epoch % 1 == 0:
            losses_5.append((num_examples_seen, np.mean(losses)))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if _args.verbose:
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                      (time, num_examples_seen, epoch, np.mean(losses)))
            sys.stdout.flush()
            prediction = []
            # 因为每棵树都不同，所以测试的训练和测试的batch都为1；后续有待改进
            for j in range(len(y_test)):
                prediction.append(
                    model.predict_up(torch.Tensor(x_test[j]).cuda(device).long())
                        .cpu().data.numpy().tolist())
            res = evaluation_4class(prediction, y_test)
            highest_acc = max(highest_acc, res[1])
            if _args.verbose:
                print(res)
                print()
            sys.stdout.flush()
        sys.stdout.flush()
        losses = []
    print('最高acc：', highest_acc)
    print('#' * 80)


if __name__ == '__main__':
    main()
