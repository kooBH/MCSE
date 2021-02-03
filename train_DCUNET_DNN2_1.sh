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
python ./src/trainerDCUNET.py -c ./config/DCUNET_t3.yaml -v DCUNET_t3
