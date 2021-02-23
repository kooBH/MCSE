#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1

## 2021-02-09
#TARGET=DCUNET_t4
#python ./src/trainerDCUNET2ch.py -c ./config/${TARGET}.yaml -v ${TARGET}_2ch -d cuda:0 -s 2780 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/${TARGET}_2ch/bestmodel.pt

## 2021-02-19
#TARGET=DCUNET_t6
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d cuda:0

## 2021-02-23
TARGET=DCUNET_t7
python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d cuda:0
