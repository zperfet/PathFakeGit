#!/usr/bin/env bash

#highest=()
for ((seed = 100; seed < 110; ++seed)); do
  echo '处理种子:'$seed
  python MainResponseWAE.py --seed $seed --cuda 0 --novae 1 --norandom 0 --train_threshold 200 --verbose 1 --dataset 15 --rate_lambda 0.5
done

#twitter16: 189_0.068 twitter15:
