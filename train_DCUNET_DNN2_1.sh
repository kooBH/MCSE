#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


## 2021-01-28
#python ./src/trainerDCUNET.py -c ./config/DNN2_DCUNET_t1.yaml -v DCUNET_t1 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/t1/bestmodel.pt
#python ./src/trainerDCUNET.py -c ./config/kiosk_DCUNET_t1.yaml -v DCUNET_t1 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t1/bestmodel.pt -s 42000

## 2021-01-31
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t2.yaml -v DCUNET_t2 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t1/bestmodel.pt 

## 2021-02-01
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t2.yaml -v DCUNET_t2 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t2/bestmodel.pt -s 29750

# 2021-02-03
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 39750 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-02-05
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t5.yaml -v DCUNET_t5 

## 2021-02-06
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 69500 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-02-07
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 99250  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-02-08
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 129000  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 132100  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt

## 2021-02-09
python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3 -s 191600  --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt
