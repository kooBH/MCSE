import torch
import torch.nn.functional as F
import numpy as np
import argparse
import torchaudio
import os,glob
from model.Unet20 import Unet20

from utils.hparams import HParam
#from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
parser.add_argument('-o','--output_dir',type=str,required=True)
parser.add_argument('-i','--dir_target',type=str,required=True)
parser.add_argument('-d','--device',type=str,default='cuda:0')
parser.add_argument('-n','--num_process',type=int,default=4)
args = parser.parse_args()

## Parameters 
hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

list_target = glob.glob(os.path.join(args.dir_target,"noisy","*.pt"))

device = args.device
torch.cuda.set_device(device)

num_epochs = 1
batch_size = 1
test_model = args.model
win_len = hp.audio.frame
n_fft = hp.audio.frame

output_dir = args.output_dir

window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)

model = None
# Set Model
if hp.model.UNET.type == 'Unet20' : 
    print("model : Unet20")
    model = Unet20(hp).to(device)

model.load_state_dict(torch.load(test_model,map_location=device))
model.share_memory()
model.eval()

#root = "/home/data/kbh/MCSE/CGMM_RLS_MPDR/test/SNR-7/"
# target = "M03_051C010J_BTH"

def inference(ran):
    print('inference range : ' + str(ran[0]) + ' ~ ' + str(ran[-1]) )
    for idx in ran :
        path_noisy = list_target[idx]
        name = path_noisy.split('/')[-1]
        id = name.split('.')[0]

        noisy = torch.load(os.path.join(path_noisy))
        estim = torch.load(os.path.join(args.dir_target,"estimated_speech",name))
        noise = torch.load(os.path.join(args.dir_target,"estimated_noise",name))

        length = noisy.shape[1]
        ## length must be multiply of 16
        target_length = int(16*np.floor(length/16)+16)

        if length < target_length : 
            need = target_length - length
            noisy =  F.pad(noisy,(0,0,0,need,0,0),'constant',value=0)
            noise =  F.pad(noise,(0,0,0,need,0,0),'constant',value=0)
            estim =  F.pad(estim,(0,0,0,need,0,0),'constant',value=0)
        wav_input = torch.istft(estim.to(device), n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)
        wav_input = torch.unsqueeze(wav_input,0).to("cpu")

        phase = torch.angle(estim[:,:,0] + estim[:,:,1]*1j)
        noisy = torch.sqrt(noisy[:,:,0]**2 + noisy[:,:,1]**2)
        estim = torch.sqrt(estim[:,:,0]**2 + estim[:,:,1]**2)
        noise = torch.sqrt(noise[:,:,0]**2 + noise[:,:,1]**2)

        input = torch.stack((estim,noisy,noise),0)

        with torch.no_grad():
            input = torch.unsqueeze(input,0).to(device)
            phase = torch.unsqueeze(phase,0).to(device)

            mask = model(input)
            # [B, (noisy,estim,noise), F, T, Cplx]
            mag_output = input[:, 0, :, :] * mask
            cplx_output = mag_output*torch.exp(phase.to(device)*1j)

            wav_output = torch.istft(cplx_output, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

            wav_output = wav_output.to('cpu')
            ## Normalize
            max_val = torch.max(torch.abs(wav_output))
            wav_output= wav_output/max_val

            ## Save
            torchaudio.save(output_dir+'/'+'input/'+str(id)+'.wav',src=wav_input,sample_rate=hp.audio.samplerate,bits_per_sample=16)
            torchaudio.save(output_dir+'/'+'output/'+str(id)+'.wav',src=wav_output[:,:],sample_rate=hp.audio.samplerate,bits_per_sample=16)

if __name__ == '__main__':

    ## Multi Processing
    # save 8 threads for others
    set_start_method('spawn')

    num_process = args.num_process

    processes = []
    ran = np.array_split(range(len(list_target)),num_process) 
    #print(np.shape(ran))

    os.makedirs(output_dir+'/input',exist_ok=True)
    os.makedirs(output_dir+'/output',exist_ok=True)

    for rank in range(num_process):
        p = mp.Process(target=inference, args=(ran[rank][:],) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


