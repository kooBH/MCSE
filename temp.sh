#!/bin/bash

TARGET=test
python ./src/trainerDCUNET.py -c ./config/${TARGET}.yaml -v ${TARGET} --device 'cuda:1'
