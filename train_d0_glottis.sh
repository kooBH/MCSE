#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1

#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' --chkpt /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -s 12450

## 2021-09-08
#TARGET=UNET_t6
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 

## 2021-09-10
#TARGET=UNET_t11
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 

## 2021-09-10
#TARGET=UNET_t14
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 


## 2021-09-13
#TARGET=UNET_t16
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0' 

## 2021-09-13
TARGET=UNET_t22
python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:0' 
