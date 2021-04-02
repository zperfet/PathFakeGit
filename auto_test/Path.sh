#!/usr/bin/env bash

#highest=()
for (( seed = 100; seed < 110; ++seed )); do
    echo '处理种子:'$seed
    python MainPathVoting.py --seed $seed --cuda 1 --novae 1 --norandom 0 --train_threshold 200 --epoch 150 --encoder_index 100_0.143 --lr 0.05 --verbose 1 --load_wae_encoder 1
#    echo 获取精度:$?
#    highest+=($?)
#    echo 全部精度:
#    for i in "${highest[*]}"; do echo $i; done
#    echo
done
