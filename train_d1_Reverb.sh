#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-02-09
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt


## 2021-09-03
#TARGET=UNET_t2
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 


## 2021-09-03
#TARGET=UNET_t4
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 

## 2021-09-09
#TARGET=UNET_t9
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 

## 2021-09-11
#TARGET=UNET_t13
#python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 

## 2021-09-13
TARGET=UNET_t21
python ./src/trainerUNET.py -c ./config/UNET//${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 
