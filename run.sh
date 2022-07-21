#!/bin/bash

set -e

# train cifar10
time python3 train_ofa_net_cifar10.py \
    --task kernel \
    | tee log/ofa_net_cifar10_kernel.log
time python3 train_ofa_net_cifar10.py \
    --task depth \
    --phase 1 \
    | tee log/ofa_net_cifar10_depth_1.log
time python3 train_ofa_net_cifar10.py \
    --task depth \
    --phase 2 \
    | tee log/ofa_net_cifar10_depth_2.log
time python3 train_ofa_net_cifar10.py \
    --task expand \
    --phase 1 \
    | tee log/ofa_net_cifar10_expand_1.log
time python3 train_ofa_net_cifar10.py \
    --task expand \
    --phase 2 \
    | tee log/ofa_net_cifar10_expand_2.log
