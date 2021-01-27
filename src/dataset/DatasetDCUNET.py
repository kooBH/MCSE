import os, glob
import torch
import librosa
import numpy as np

class DatasetDCUNET(torch.utils.data.Dataset):
    def __init__(self, stft_root,wav_root,target, form,num_frame=41):
        self.stft_root = stft_root
        self.wav_root = wav_root
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
#        print('['+str(index)+'] : ' + path)

        npy_noisy = np.load(self.stft_root+'/'+'noisy'+'/'+self.data_list[index]+'.npy')
        npy_noise = np.load(self.stft_root+'/'+'noise'+'/'+self.data_list[index]+'.npy')
        npy_estim = np.load(self.stft_root+'/'+'estim'+'/'+self.data_list[index]+'.npy')

        npy_wav_clean,sr = librosa.load(self.wav_root+'/'+'clean'+'/'+self.data_list[index]+'.wav',sr=16000)
        npy_wav_noisy,sr = librosa.load(self.wav_root+'/'+'noisy'+'/'+self.data_list[index]+'.wav',sr=16000)

        ## sampling routine ##

        # [Freq, Time, complex] 
        length = np.size(npy_noisy, 1)
        over = length - self.num_frame
        need = -over

        start = 0

        if over >= 0 :
            start = np.random.randint(low=0,high=over)

            npy_noisy = npy_noisy[:,start:start+self.num_frame,:]
            npy_noise = npy_noise[:,start:start+self.num_frame,:]
            npy_estim = npy_estim[:,start:start+self.num_frame,:]

            # [samples]
            npy_wav_clean = npy_wav_clean[start*256:start*256 + self.num_frame*256]
            npy_wav_noisy = npy_wav_noisy[start*256:start*256 + self.num_frame*256]
        # zero-padding
        else :
            npy_noisy =  np.pad(npy_noisy,((0,0),(0,need),(0,0)),'constant',constant_values=0)
            npy_noise =  np.pad(npy_noise,((0,0),(0,need),(0,0)),'constant',constant_values=0)
            npy_estim =  np.pad(npy_estim,((0,0),(0,need),(0,0)),'constant',constant_values=0)
            
            npy_wav_clean = npy_wav_clean[:length*256]
            npy_wav_noisy = npy_wav_noisy[:length*256]

            npy_wav_clean = np.pad(npy_wav_clean,(0,need*256),'constant',constant_values=0)
            npy_wav_noisy = np.pad(npy_wav_noisy,(0,need*256),'constant',constant_values=0)

        torch_noisy = torch.from_numpy(npy_noisy)
        torch_noise = torch.from_numpy(npy_noise)
        torch_estim = torch.from_numpy(npy_estim)

        torch_wav_clean = torch.from_numpy(npy_wav_clean)
        torch_wav_noisy = torch.from_numpy(npy_wav_noisy)

        """
        input : 3-channel (noisy,estim,clean)
        loss : weighted SDR loss in time domian
        label : clean wav, noisy wav
        """

        data = {"input":torch.stack((torch_noisy,torch_estim,torch_noise),0), "clean":torch_wav_clean,"noisy":torch_wav_noisy}

        return data

    def __len__(self):
        return len(self.data_list)
