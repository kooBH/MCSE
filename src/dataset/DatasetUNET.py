import os, glob
import torch
import librosa
import numpy as np
import torch.nn.functional as F

def biquad_filter(x,r=None):
    if r is None:
        r = (np.random.rand(4)-0.5)*(3/8)

    y = torch.zeros(x.shape)
    
    n_fft = x.shape[0]
    length = x.shape[1]
    
    y = torch.zeros((n_fft,length,2),dtype=torch.complex64)
     
    y[0] = x[0]
    y[1] = x[1] + r[0]*x[0] - r[2]*y[0]
    y[:,:,1] = x[:,:,1]
    
    for i in range(2,length):
        y[:,i,0] = x[:,i,0] + r[0]*x[:,i-1,0] + r[1]*x[:,i-2,0] - r[2]*y[:,i-1,0] - r[3]*y[:,i-2,0]


class DatasetUNET(torch.utils.data.Dataset):
    def __init__(self, hp,is_train=True):
        self.hp = hp 
        if is_train : 
            self.root = os.path.join(hp.data.root,'train')
        else :
            self.root = os.path.join(hp.data.root,'test')

        print('root : ' + self.root)

        self.num_frame = hp.model.UNET.num_frame
        self.SNRs = hp.data.SNR
        self.target = self.root.split('/')[-2]

        if self.target in ['CGMM_RLS_MPDR','CGMM_RLS_MPDR_norm_2','AuxIVA_DC_SVE','WPE_MLDR_OMLSA'] : 
            pass
        else :
            raise Exception('unsupported target ' + str(self.target) )

        if type(self.SNRs) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(root,SNRs, 'noisy','*.pt'), recursive=False)]
        elif type(self.SNRs) == list : 
            self.data_list = []
            for i in self.SNRs :  
                self.data_list = self.data_list + [x for x in glob.glob(os.path.join(self.root, i,'noisy' ,'*.pt'), recursive=False)]
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
        if  self.target == 'CGMM_RLS_MPDR_norm_2' or self.target == 'WPE_MLDR_OMLSA': 
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


        if self.hp.augment.type == 'biquad':
            r = (np.random.rand(4)-0.5)*(3/8)
            noisy = biquad_filter(noisy,r)
            estim = biquad_filter(estim,r)
            noise = biquad_filter(noise,r)
        elif self.hp.augment.type == 'none' : 
            pass
        else :
            raise Exception('Unknown augmentation method : ' + str(self.hp.augment.type))

        phase = None
        if self.hp.model.UNET.input == 'noisy' : 
            phase_input = torch.angle(noisy[:,:,0] + noisy[:,:,1]*1j)
            phase_clean = torch.angle(clean[:,:,0] + clean[:,:,1]*1j)
        # estim
        else :
            phase_input = torch.angle(estim[:,:,0] + estim[:,:,1]*1j)
            phase_clean = torch.angle(clean[:,:,0] + clean[:,:,1]*1j)

        noisy = torch.sqrt(noisy[:,:,0]**2 + noisy[:,:,1]**2)
        estim = torch.sqrt(estim[:,:,0]**2 + estim[:,:,1]**2)
        noise = torch.sqrt(noise[:,:,0]**2 + noise[:,:,1]**2)
        clean = torch.sqrt(clean[:,:,0]**2 + clean[:,:,1]**2)

        if self.hp.model.UNET.input == 'noisy' : 
            input = torch.stack((noisy,estim,noise),0)
        else :
            input = torch.stack((estim,noisy,noise),0)

        data = {"input":input, "clean":clean,'phase':torch.stack((phase_input,phase_clean),0)}
        return data

    def __len__(self):
        return len(self.data_list)
