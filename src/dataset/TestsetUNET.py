import os, glob
import torch
import librosa
import numpy as np

class TestsetUNET(torch.utils.data.Dataset):
    def __init__(self, hp,stft_root,target,form):
        self.stft_root = stft_root
        self.hp = hp

        if type(target) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(stft_root+'/noisy/', target, form), recursive=False) if not os.path.isdir(x)]
        elif type(target) == list : 
            self.data_list = []
            for i in target : 
                self.data_list = self.data_list + [x for x in glob.glob(os.path.join(stft_root+'/noisy/', i, form), recursive=False) if not os.path.isdir(x)]
        else : 
            raise Exception('Unsupported type for target')

        # Extract id only.
        for i in range(len(self.data_list)) : 
            tmp = self.data_list[i]
            tmp = tmp.split('/')
            self.data_list[i] = tmp[-2] + '/' + tmp[-1]
            self.data_list[i] = (self.data_list[i].split('.'))[0]

    def __getitem__(self, index):
        path = self.data_list[index]

        tmp = path.split('/')
        data_dir = tmp[0]
        data_name = tmp[1]
#        print('['+str(index)+'] : ' + path)

        npy_noisy = np.load(self.stft_root+'/'+'noisy'+'/'+self.data_list[index]+'.npy')
        npy_noise = np.load(self.stft_root+'/'+'noise'+'/'+self.data_list[index]+'.npy')
        npy_estim = np.load(self.stft_root+'/'+'estim'+'/'+self.data_list[index]+'.npy')

        # [Freq, Time, complex] 
        length = np.size(npy_noisy,1)

        ## zero-padding

        ## length must be multiply of 16
        target_length = int(16*np.floor(length/16)+16)

        if length < target_length : 
            need = target_length - length
            npy_noisy =  np.pad(npy_noisy,((0,0),(0,need),(0,0)),'constant',constant_values=0)
            npy_noise =  np.pad(npy_noise,((0,0),(0,need),(0,0)),'constant',constant_values=0)
            npy_estim =  np.pad(npy_estim,((0,0),(0,need),(0,0)),'constant',constant_values=0)

        noisy = torch.from_numpy(npy_noisy)
        noise = torch.from_numpy(npy_noise)
        estim = torch.from_numpy(npy_estim)

        phase = None
        if self.hp.model.UNET.input == 'noisy' : 
            phase = torch.angle(noisy[:,:,0] + noisy[:,:,1]*1j)
        # estim
        else :
            phase = torch.angle(estim[:,:,0] + estim[:,:,1]*1j)

        noisy = torch.sqrt(noisy[:,:,0]**2 + noisy[:,:,1]**2)
        estim = torch.sqrt(estim[:,:,0]**2 + estim[:,:,1]**2)
        noise = torch.sqrt(noise[:,:,0]**2 + noise[:,:,1]**2)

        if self.hp.model.UNET.input == 'noisy' : 
            input = torch.stack((noisy,estim,noise),0)
        else :
            input = torch.stack((estim,noisy,noise),0)

        data = {"input":input,"phase":phase}

        return data, target_length, data_dir, data_name

    def __len__(self):
        return len(self.data_list)
