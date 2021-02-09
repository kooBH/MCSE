#/bin/bash

## 2021-02-02
#python src/inferenceDCUNET.py  -c config/DCUNET_t2.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t2/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t2

## 2021-02-05
#python src/inferenceDCUNET.py  -c config/DCUNET_t3.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t3

#python src/inferenceDCUNET.py  -c config/DCUNET_t4.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t4/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t4

#python src/inferenceDCUNET.py  -c config/DCUNET_t1.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t1/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t1_mag -t magnitude

#python src/inferenceDCUNET.py  -c config/DCUNET_t3.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t3/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t3_mag -t magnitude

## 2021-02-06
python src/inferenceDCUNET.py  -c config/DCUNET_t1.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/DCUNET_t1/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/DCUNET_t1_mag -t magnitude_with_estim_phase