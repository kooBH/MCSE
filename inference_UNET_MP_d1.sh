#/bin/bash

## 2021-05-16
#TARGET=DCUNET_t21
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/model_step_3769.pt -o /home/nas/user/kbh/MCSE/output/${TARGET}_s3769 -d cuda:0
#python src/inferenceDCUNET.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/MCSE/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE/output/${TARGET} -d cuda:0


## 2021-09-09
#TARGET=UNET_t3
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET}_MP -d cuda:1 -n 4
#TARGET=UNET_t4
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET}_MP -d cuda:1 -n 4
#TARGET=UNET_t1
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET}_MP -d cuda:1 -n 4


## 2021-09-09
#TARGET=UNET_t8
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8


## 2021-09-11
#TARGET=UNET_t12
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8


## 2021-09-13
#TARGET=UNET_t14
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

#TARGET=UNET_t15
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

#TARGET=UNET_t5
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

#TARGET=UNET_t7
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

#TARGET=UNET_t10
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

#TARGET=UNET_t13
#python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

## 2021-09-14
TARGET=UNET_t20
python src/inferenceUNET_MP.py  -c config/UNET/${TARGET}.yaml -m /home/nas/user/kbh/MCSE_UNET/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/MCSE_UNET/output/${TARGET} -d cuda:1 -n 8

