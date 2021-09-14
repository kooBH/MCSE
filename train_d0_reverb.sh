#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-02-09
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-09-03
#TARGET=UNET_t1
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 


## 2021-09-03
#TARGET=UNET_t3
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 


## 2021-09-09
#TARGET=UNET_t6
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 

## 2021-09-10
#TARGET=UNET_t10
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 

## 2021-09-13
TARGET=UNET_t19
python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0' 


