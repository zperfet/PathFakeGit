#!/usr/bin/env bash

#highest=()
for (( seed = 100; seed < 120; ++seed )); do
    echo '处理种子:'$seed
    python Main_TD_RvNN_Ma.py --seed $seed --cuda 2
#    echo 获取精度:$?
#    highest+=($?)
#    echo 全部精度:
#    for i in "${highest[*]}"; do echo $i; done
#    echo
done
