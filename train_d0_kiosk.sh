#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-02-09
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt


## 2021-05-04
#TARGET=DCUNET_t15
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' -s 28130 --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt

## 2021-05-06
#TARGET=DCUNET_t16
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

#TARGET=DCUNET_t17
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:0' -s 5140 --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt

## 2021-05-10
#TARGET=DCUNET_t17
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -s 31040

## 2021-05-12
#TARGET=DCUNET_t16
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -s 97560

## 2021-05-15
#TARGET=DCUNET_t20
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' --chkpt /home/nas/user/kbh/MCSE/chkpt/DCUNET_t16/bestmodel.pt -s 0

## 2021-05-20
#TARGET=DCUNET_t21
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' --chkpt /home/nas/user/kbh/MCSE/chkpt/DCUNET_t16/bestmodel.pt -s 42980

## 2021-09-03
#TARGET=UNET_t3
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0' 

## 2021-09-09
#TARGET=UNET_t5
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0' 

## 2021-09-11
#TARGET=UNET_t5
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0'  --chkpt /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt
# -s 1130


## 2021-09-13
TARGET=UNET_t17
python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0' 