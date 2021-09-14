#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-02-09
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-08-17
#TARGET=DCUNET_t46
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 
#TARGET=DCUNET_t47
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 
#TARGET=DCUNET_t48
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 
#TARGET=DCUNET_t49
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1' 


## 2021-08-17
#TARGET=DCUNET_t46
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt --device 'cuda:1' -s 5340
#TARGET=DCUNET_t47
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt --device 'cuda:1' -s 5650

## 2021-08-18
#TARGET=DCUNET_t47
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt --device 'cuda:1' -s 6270
#TARGET=DCUNET_t48
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt --device 'cuda:1' -s 9420

## 2021-08-20
#TARGET=DCUNET_t51
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-08-21
#TARGET=DCUNET_t52
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-08-22
#TARGET=DCUNET_t55
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'
#TARGET=DCUNET_t56
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'
#TARGET=DCUNET_t57
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-08-24
#TARGET=DCUNET_t58
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-08-25
#TARGET=DCUNET_t59
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'

## 2021-08-25
#TARGET=DCUNET_t46
#python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --chkpt /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt --device 'cuda:1' -s 22300


## 2021-08-25
#TARGET=UNET_t0
#python ./src/trainerDCUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1' 0


## 2021-09-02
#TARGET=UNET_t2
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:0'

## 2021-09-07
#TARGET=UNET_t0
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 

## 2021-09-08
#TARGET=UNET_t0
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' --chkpt /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -s 12450

## 2021-09-09
#TARGET=UNET_t8
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 

## 2021-09-09
#TARGET=UNET_t12
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 

## 2021-09-11
#TARGET=UNET_t15
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 

## 2021-09-13
#TARGET=UNET_t20
#python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 

## 2021-09-14
TARGET=UNET_t23
python ./src/trainerUNET.py -c ./config/UNET/${TARGET}.yaml -v ${TARGET} -d 'cuda:1' 
