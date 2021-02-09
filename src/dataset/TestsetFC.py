import os, glob
import torch
import librosa
import numpy as np

class TestsetFC(torch.utils.data.Dataset):
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

        temp = path.split('/')
        data_dir = temp[0]
        data_name = temp[1]

        npy_noisy = np.load(self.stft_root+'/'+'noisy'+'/'+self.data_list[index]+'.npy')
        npy_noise = np.load(self.stft_root+'/'+'noise'+'/'+self.data_list[index]+'.npy')
        npy_estim = np.load(self.stft_root+'/'+'estim'+'/'+self.data_list[index]+'.npy')

        target_length = np.size(npy_noisy,1)  

        # Since single frame output
        torch_noisy = torch.from_numpy(npy_noisy)
        torch_noise = torch.from_numpy(npy_noise)
        torch_estim = torch.from_numpy(npy_estim)

        """
        input : flat data for 2*length +1 frame complex(noisy + estim + noise )
        """
        torch_input = torch.stack((torch_noisy,torch_estim,torch_noise),1)

        data = {"input":torch_input,"length":target_length,"dir":data_dir,"name":data_name}
        return data

    def __len__(self):
        return len(self.data_list)