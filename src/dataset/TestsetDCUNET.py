import os, glob
import torch
import librosa
import numpy as np

class TestsetDCUNET(torch.utils.data.Dataset):
    def __init__(self, stft_root,target, form,num_frame=80):
        self.stft_root = stft_root
        self.num_frame = num_frame

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



        torch_noisy = torch.from_numpy(npy_noisy)
        torch_noise = torch.from_numpy(npy_noise)
        torch_estim = torch.from_numpy(npy_estim)

        data = torch.stack((torch_noisy,torch_estim,torch_noise),0)

        return data, target_length, data_dir, data_name

    def __len__(self):
        return len(self.data_list)
