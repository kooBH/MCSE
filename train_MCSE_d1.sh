#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-02-09
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt


## 2021-05-04
#TARGET=DCUNET_t15
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' -s 28130 --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt

## 2021-05-06
TARGET=DCUNET_t16
python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

