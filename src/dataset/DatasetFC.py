import os, glob
import torch
import librosa
import numpy as np

class DatasetFC(torch.utils.data.Dataset):
    def __init__(self, stft_root,target, form,block=3):
        self.stft_root = stft_root
        self.block = block

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

        npy_noisy = np.load(self.stft_root+'/'+'noisy'+'/'+self.data_list[index]+'.npy')
        npy_noise = np.load(self.stft_root+'/'+'noise'+'/'+self.data_list[index]+'.npy')
        npy_estim = np.load(self.stft_root+'/'+'estim'+'/'+self.data_list[index]+'.npy')
        npy_clean = np.load(self.stft_root+'/'+'clean'+'/'+self.data_list[index]+'.npy')

        ## sampling routine ##

        # [Freq, Time, complex] 
        length = np.size(npy_noisy,1)  
        # rand on start index  
        length = length - (2*self.block + 1)
        start = np.random.randint(length)

        npy_noisy = npy_noisy[:,start:start+(2*self.block+1),:]
        npy_noise = npy_noise[:,start:start+(2*self.block+1),:]
        npy_estim = npy_estim[:,start:start+(2*self.block+1),:]
        npy_clean = npy_clean[:,start+self.block,:]

        # Since single frame output

        torch_noisy = torch.from_numpy(npy_noisy)
        torch_noise = torch.from_numpy(npy_noise)
        torch_estim = torch.from_numpy(npy_estim)
        torch_clean = torch.from_numpy(npy_clean)

        """
        input : flat data for 2*length +1 frame complex(noisy + estim + noise )
        target : 1 frame complex (clean)
        """
        torch_input = torch.stack((torch_noisy,torch_estim,torch_noise),1)
        torch_input = torch.reshape(torch_input,(513*3*(2*self.block+1),2))

        data = {"input":torch_input, "target":torch_clean}
        return data

    def __len__(self):
        return len(self.data_list)