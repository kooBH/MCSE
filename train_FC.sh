#!/bin/bash



#python ./src/trainerFC.py -c ./config/DNN2_FC_t1.yaml -v FC_t1

## 2021-01-28
#python ./src/trainerFC.py -c ./config/kiosk_FC_t1.yaml -v FC_t1 --chkpt 
#python ./src/trainerFC.py -c ./config/kiosk_FC_t1.yaml -v FC_t1 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/FC_t1/bestmodel.pt -s 13140

#python ./src/trainerFC.py -c ./config/kiosk_FC_t2.yaml -v FC_t2
# python ./src/trainerFC.py -c ./config/kiosk_FC_t2.yaml -v FC_t2 --chkpt /home/nas/user/kbh/3-channel-dnn/chkpt/FC_t2/bestmodel.pt -s 9220

## 2021-01-29
python ./src/trainerFC.py -c ./config/kiosk_FC_t3.yaml -v FC_t3
