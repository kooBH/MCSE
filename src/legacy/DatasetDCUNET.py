import os, glob
import torch
import librosa
import numpy as np
import torch.nn.functional as F

class DatasetDCUNET(torch.utils.data.Dataset):
    def __init__(self, root,SNRs, num_frame=80):

        self.root = root
        self.num_frame = num_frame
        self.SNRs = SNRs
        self.target = root.split('/')[-1]

        if type(SNRs) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(root,SNRs, 'noisy','*.pt'), recursive=False)]
        elif type(SNRs) == list : 
            self.data_list = []
            for i in SNRs :  
                self.data_list = self.data_list + [x for x in glob.glob(os.path.join(root, i,'noisy' ,'*.pt'), recursive=False)]
        else : 
            raise Exception('Unsupported type for target')

    def __getitem__(self, index):
        path = self.data_list[index]
#        print('['+str(index)+'] : ' + path)
        file_name = path.split('/')[-1]
        SNR = path.split('/')[-3]

        root = self.root

        noisy = torch.load(os.path.join(root,SNR,'noisy')+'/'+file_name)
        estim= torch.load(os.path.join(root,SNR,'estimated_speech')+'/'+file_name)
        noise = torch.load(os.path.join(root,SNR,'estimated_noise')+'/'+file_name)
        if  self.target == 'CGMM_RLS_MPDR_norm_2' : 
            clean = torch.load(os.path.join(root,SNR,'clean')+'/'+file_name)
        else  :
            clean = torch.load(os.path.join(root,'clean')+'/'+file_name)

       ## sampling routine ##

        # [Freq, Time, complex] 
        length = noisy.shape[1]
        need = self.num_frame - length

        start_idx = np.random.randint(low=0,high = length)

        # padding on head 
        if start_idx + self.num_frame > length :  
            shortage = (start_idx + self.num_frame) - length
            noisy = noisy[:,start_idx:,:]
            noisy = F.pad(noisy,(0,0,shortage,0,0,0),'constant',value=0)
            estim = estim[:,start_idx:,:]
            estim = F.pad(estim,(0,0,shortage,0,0,0),'constant',value=0)
            noise = noise[:,start_idx:,:]
            noise = F.pad(noise,(0,0,shortage,0,0,0),'constant',value=0)
            clean = clean[:,start_idx:,:]
            clean = F.pad(clean,(0,0,shortage,0,0,0),'constant',value=0)
            #print(str(1) +  ' ' + str(start_idx)+ '| '+str(length) + '|'+str(shortage) + '|' + str(noisy.shape))
        # padding on tail
        elif start_idx >= length - self.num_frame : 
            shortage = start_idx - length + self.num_frame 
            noisy =  noisy[:,start_idx:,]
            noisy = F.pad(noisy,(0,0,0,shortage,0,0),'constant',value=0)
            estim =  estim[:,start_idx:,]
            estim = F.pad(estim,(0,0,0,shortage,0,0),'constant',value=0)
            noise =  noise[:,start_idx:,]
            noise = F.pad(noise,(0,0,0,shortage,0,0),'constant',value=0)
            clean =  clean[:,start_idx:,]
            clean = F.pad(clean,(0,0,0,shortage,0,0),'constant',value=0)
            #print(str(2) +  ' ' + str(start_idx)+ '| '+str(length) + '|'+str(shortage) + '|' + str(noisy.shape))
        else :
            noisy =  noisy[:,start_idx:start_idx+self.num_frame,:]
            estim =  estim[:,start_idx:start_idx+self.num_frame,:]
            noise =  noise[:,start_idx:start_idx+self.num_frame,:]
            clean =  clean[:,start_idx:start_idx+self.num_frame,:]
            #print(str(3) +  ' ' + str(start_idx)+ '| '+str(length)+'|'+str(noisy.shape))
        data = {"input":torch.stack((noisy,estim,noise),0), "clean":clean}
        return data

    def __len__(self):
        return len(self.data_list)
