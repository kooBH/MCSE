#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1

## 2021-09-03
#TARGET=UNET_t4
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 

## 2021-09-07
#TARGET=UNET_t1
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-09-09
#TARGET=UNET_t7
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-09-11
#TARGET=UNET_t7
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' --chkpt /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -s 1130

## 2021-09-13
TARGET=UNET_t18
python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 
