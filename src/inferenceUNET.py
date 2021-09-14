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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-o','--output_dir',type=str,required=True)
    parser.add_argument('-d','--device',type=str,default='cuda:0')
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

    model = None
    # Set Model
    if hp.model.UNET.type == 'Unet20' : 
        print("model : Unet20")
        model = Unet20(input_channels = hp.model.UNET.channels).to(device)
    elif hp.model.UNET.type == 'TRU':
        print("model : UNET Tiny Recurrent Unet")
        model = UNET().to(device)
    else :
        raise Exception('No model such as '+ str(hp.model.UNET.type))
    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    
    print('NOTE::Loading pre-trained model : ' + test_model)

    n_fft = hp.audio.frame

    window = torch.hann_window(window_length=n_fft,periodic=True, dtype=None, layout=torch.strided, device=device, requires_grad=False)

    ## Inference
    with torch.no_grad():
        for i, (data, length, data_dir, data_name) in enumerate(tqdm(test_loader)):
            input = data["input"][:,:hp.model.UNET.channels,:,:].to(device)
            phase = data["phase"].to(device)

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
            torchaudio.save(output_dir+'/'+str(data_dir[0])+'/'+str(data_name[0])+'.wav',src=wav_output[:,:],sample_rate=hp.audio.samplerate,bits_per_sample=16)
