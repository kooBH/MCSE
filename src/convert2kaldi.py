import torch
import argparse
import torchaudio
import os
import numpy as np
from kaldiio import WriteHelper

from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.FC import FC
from model.SKDAE import SKDAE

import dataset.Dataset as data

from utils.hparams import HParam
from utils.writer import MyWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath','-i',type=str,required=True)
    parser.add_argument('--outpath','-o',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    ## Parameters
    device = args.device
    #torch.cuda.set_device(device)
    batch_size = 1 
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    ## Data
    list_dt05_simu = ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu']
    list_dt05_real = ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real']
    list_et05_simu = ['et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
    list_et05_real = ['et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real']

    list_dirs = [list_dt05_simu,list_dt05_real,list_et05_simu,list_et05_real]

    version = 'some'

    writer = None

    ##  params
    cnt = 0
    idx_category = 0
    list_category= [
        'dt05_simu',
        'dt05_real',
        'et05_simu',
        'et05_real'
    ]
    # Need to create ${nj_MFCC} of ark,scp pairs
    nj_MFCC = 8

    ## Inference
    for i in list_dirs :
        len_dataset = len(i)
        cur_category = list_category[idx_category]
        idx_category+=1
        max_item = int(len_dataset / nj_MFCC)
        if len_dataset % nj_MFCC != 0 :
            raise Exception('len_dataset must be devided by nj')
        ver = 1
        cnt = 0

        for j, wav in enumerate(tqdm(i,desc=cur_category)):
            #output = torchaudio.compliance.kaldi.mfcc(wav)
            # ark,scp for each category
            # after 250 samples, new ark,scp
            if cnt == 0 :
                filename= 'ark,scp:'+args.outpath+'/raw_mfcc_'+cur_category+'_'+version+'.'+str(ver)+'.ark,'+args.outpath+'/raw_mfcc_'+cur_category+'_'+version+'.'+str(ver)+'.scp'
                writer= WriteHelper(filename,compression_method=2)
            #writer(name,output)
            print(cur_gategory + ' | ' +str(ver)+' | '+ filename)

            cnt += 1
            if cnt> max_item :
                cnt = 0
                ver +=1