#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1


python ./src/trainerDCUNET.py -c ./config/DNN2_DCUNET_t1.yaml -v DCUNET_t1 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/t1/bestmodel.pt
