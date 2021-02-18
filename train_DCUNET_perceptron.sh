#!/bin/bash

# default
#python ./src/trainer_DCUNET.py -c ./config/default.yaml -v t1

## 2021-02-09
TARGET=DCUNET_t4
python ./src/trainerDCUNET2ch.py -c ./config/${TARGET}.yaml -v ${TARGET}_2ch
