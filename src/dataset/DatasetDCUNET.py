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

        if type(SNRs) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(root,target, 'noisy','*.pt'), recursive=False)]
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

        noisy = torch.load(os.path.join(root,SNR,'noisy',)+'/'+file_name)
        estim= torch.load(os.path.join(root,SNR,'estimated_speech',)+'/'+file_name)
        noise = torch.load(os.path.join(root,SNR,'estimated_noise',)+'/'+file_name)
        clean = torch.load(os.path.join(root,'clean',)+'/'+file_name)

       ## sampling routine ##

        # [Freq, Time, complex] 
        length = noisy.shape[1]
        need = self.num_frame - length

        start = 0

        if need <= 0 :
            if need != 0:
                start = np.random.randint(low=0,high=-need)
            else :
                start = 0
            noisy = noisy[:,start:start+self.num_frame,:]
            noise = noise[:,start:start+self.num_frame,:]
            estim = estim[:,start:start+self.num_frame,:]
            clean = clean[:,start:start+self.num_frame,:]
        # zero-padding
        elif need > 0 :
            noisy =  F.pad(noisy,((0,0),(0,need),(0,0)),'constant',value=0)
            noise =  F.pad(noise,((0,0),(0,need),(0,0)),'constant',value=0)
            estim =  F.pad(estim,((0,0),(0,need),(0,0)),'constant',value=0)
            clean =  F.pad(clean,((0,0),(0,need),(0,0)),'constant',value=0)
            
        data = {"input":torch.stack((noisy,estim,noise),0), "clean":clean}
        return data

    def __len__(self):
        return len(self.data_list)
