#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1

# 2021-02-02
#python ./src/trainerDCUNET.py -c ./config/DCUNET_t4.yaml -v DCUNET_t4

## 2021-02-04
python ./src/trainerDCUNET.py -c ./config/DCUNET_t4.yaml -v DCUNET_t4 -s 57950 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t4/bestmodel.pt
