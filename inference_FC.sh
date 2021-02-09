#/bin/bash

## 2021-02-09
TARGET=FC_t2
python src/inferenceFC.py  -c config/${TARGET}.yaml -m /home/nas/user/kbh/3-channel-dnn/chkpt/${TARGET}/bestmodel.pt -o /home/nas/user/kbh/3-channel-dnn/output/${TARGET}
