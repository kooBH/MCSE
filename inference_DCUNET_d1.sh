#/bin/bash

## 2021-07-12
#TARGET=DCUNET_t24
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-07-14
#TARGET=DCUNET_t26
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-07-22
#TARGET=DCUNET_t28
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1
#TARGET=DCUNET_t29
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-08-05
#TARGET=DCUNET_t36
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-08-05
#TARGET=DCUNET_t38
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-08-09
#TARGET=DCUNET_t39
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1
#TARGET=DCUNET_t40
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1

## 2021-08-13
#TARGET=DCUNET_t37
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET}_mag_noisy_phase -d cuda:1 -t magnitude
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET}_mag_estim_phase -d cuda:1 -t magnitude_with_estim_phase

#TARGET=DCUNET_t43
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET}_mag_noisy_phase -d cuda:1 -t magnitude
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET}_mag_estim_phase -d cuda:1 -t magnitude_with_estim_phase

## 2021-08-18
#TARGET=DCUNET_t40
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1
#TARGET=DCUNET_t45
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1
#TARGET=DCUNET_t46
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1

## 2021-08-19
#TARGET=DCUNET_t46
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1

## 2021-08-24
#TARGET=DCUNET_t47
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1
#TARGET=DCUNET_t48
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1


## 2021-08-24
TARGET=DCUNET_t46
python src/inferenceDCUNET.py  -c config/DCUNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:1

#tmp=$(pwd)
#cd /home/kbh/kaldi/egs/chime4_mod/s5_6ch_enhan_clean
#./kbh.sh ${TARGET}
#cd $tm
