# Inferencing with Multi processor

import torch
import argparse
import numpy as np
import torchaudio
import os
from model.DCUNET import DCUNET
from model.Unet20 import Unet20

from dataset.TestsetUNET import TestsetUNET
from tqdm import tqdm
from utils.hparams import HParam
#from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
parser.add_argument('-o','--output_dir',type=str,required=True)
parser.add_argument('-d','--device',type=str,default='cuda:0')
parser.add_argument('-n','--num_process',type=int,default=4)
args = parser.parse_args()

## Parameters 
hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

device = args.device
torch.cuda.set_device(device)

num_epochs = 1
batch_size = 1
test_model = args.model
win_len = hp.audio.frame

window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)

## dirs 
list_test= ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real','et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real','dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
output_dir = args.output_dir
os.makedirs(output_dir,exist_ok=True)
for i in list_test :
    os.makedirs(os.path.join(output_dir,i),exist_ok=True)

## Dataset
test_dataset = TestsetUNET(hp,hp.data.test_root+'/STFT_R',list_test,'*.npy')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=1)

print('NOTE::test dataset loaded : ' + str(len(test_loader)))

model = None
# Set Model
if hp.model.UNET.type == 'Unet20' : 
    print("model : Unet20")
    model = Unet20(hp).to(device)
elif hp.model.UNET.type == 'TRU':
    print("model : UNET Tiny Recurrent Unet")
    model = UNET().to(device)
else :
    raise Exception('No model such as '+ str(hp.model.UNET.type))
model.load_state_dict(torch.load(test_model,map_location=device))
model.share_memory()
model.eval()

print('NOTE::Loading pre-trained model : ' + test_model)

n_fft = hp.audio.frame

def inference(ran):
    print('inference range : ' + str(ran[0]) + ' ~ ' + str(ran[-1]) )
    for idx in ran :
        #i, (data, length, data_dir, data_name ) = test_loader[idx]

        data,length,data_dir,data_name = test_dataset[idx]

        with torch.no_grad():
            input = data["input"][:hp.model.UNET.channels,:,:].to(device)
            phase = data["phase"].to(device)
            input = torch.unsqueeze(input,0)
            phase = torch.unsqueeze(phase,0)

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
            torchaudio.save(output_dir+'/'+str(data_dir)+'/'+str(data_name)+'.wav',src=wav_output[:,:],sample_rate=hp.audio.samplerate,bits_per_sample=16)

if __name__ == '__main__':

    ## Multi Processing
    # save 8 threads for others
    set_start_method('spawn')

    num_process = args.num_process

    processes = []
    ran = np.array_split(range(len(test_dataset)),num_process) 
    #print(np.shape(ran))


    for rank in range(num_process):
        p = mp.Process(target=inference, args=(ran[rank][:],) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

