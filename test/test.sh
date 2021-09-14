#!/bin/bash

VERSION=DCUNET_t28

python test.py -c ../config/${VERSION}.yaml  -m /home/nas/user/kbh/MCSE/chkpt/${VERSION}/bestmodel.pt -i /home/data/kbh/Study_Loss/ -o /home/data/kbh/Study_Loss/${VERSION} -d 'cuda:0'

