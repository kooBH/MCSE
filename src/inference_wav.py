import torch
import argparse
import numpy as np
import torchaudio
import os,glob
import librosa

from model.Unet20 import Unet20

from tqdm import tqdm
from utils.hparams import HParam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,required=True)
    parser.add_argument('-i','--dir_input',type=str,required=True)
    parser.add_argument('-o','--dir_output',type=str,required=True)
    parser.add_argument('-d','--device',type=str,default='cuda:0')
    args = parser.parse_args()

    ## Parameters 
    hp = HParam(args.config)
    print('NOTE::Loading configuration :: ' + args.config)

    device = args.device
    torch.cuda.set_device(device)

    fft_size = 1024
    
    num_epochs = 1
    batch_size = 1
    test_model = args.model
    win_len = hp.audio.frame

    window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
 
    model = None
    # Set Model
    if hp.model.UNET.type == 'Unet20' : 
        print("model : Unet20")
        model = Unet20(hp).to(device)
    else :
        raise Exception('No model such as '+ str(hp.model.UNET.type))
    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    
    print('NOTE::Loading pre-trained model : ' + test_model)

    n_fft = hp.audio.frame
    window = torch.hann_window(window_length=n_fft,periodic=True, dtype=None, layout=torch.strided, device=device, requires_grad=False)
    
    # target files 
    if not (os.path.isdir(os.path.join(args.dir_input,"noisy")) and os.path.isdir(os.path.join(args.dir_input,"estim"))) : 
        print("ERROR::input must be in \"estim\", \"noisy\" directories")
        exit(-1)
    list_target = glob.glob(os.path.join(args.dir_input,"noisy","*.wav"))
    
    os.makedirs(args.dir_output,exist_ok=True)
    
    ## Inference
    with torch.no_grad():
        for path_noisy in tqdm(list_target) :
            name_wav = path_noisy.split('/')[-1]
            noisy,sr = librosa.load(path_noisy,sr=16000)
            estim,sr = librosa.load(os.path.join(args.dir_input,'estim',name_wav),sr=16000)
            
            # Normalization
            max_val = np.max(np.abs(noisy))
            noisy = noisy / max_val
            max_val = np.max(np.abs(estim))
            estim = estim / max_val
            
            # Spectra
            spec_noisy = librosa.stft(noisy,window='hann', n_fft=fft_size, hop_length=None , win_length=None ,center=False)
            spec_estim = librosa.stft(estim,window='hann', n_fft=fft_size, hop_length=None , win_length=None ,center=False)
            
            ## length must be multiply of 16
            length = np.size(spec_noisy,1)
            target_length = int(16*np.floor(length/16)+16)

            if length < target_length : 
                need = target_length - length
                spec_noisy =  np.pad(spec_noisy,((0,0),(0,need)),'constant',constant_values=0)
                spec_estim =  np.pad(spec_estim,((0,0),(0,need)),'constant',constant_values=0)

            spec_noisy = np.complex64(spec_noisy)
            spec_estim = np.complex64(spec_estim)
            
            spec_noisy = np.concatenate((np.expand_dims(spec_noisy.real,-1),np.expand_dims(spec_noisy.imag,-1)),2)
            spec_estim = np.concatenate((np.expand_dims(spec_estim.real,-1),np.expand_dims(spec_estim.imag,-1)),2)
            
            # np -> torch
            noisy = torch.from_numpy(spec_noisy)
            estim = torch.from_numpy(spec_estim)
            
            phase = torch.angle(noisy[:,:,0] + noisy[:,:,1]*1j)
            noisy = torch.sqrt(noisy[:,:,0]**2 + noisy[:,:,1]**2)
            estim = torch.sqrt(estim[:,:,0]**2 + estim[:,:,1]**2)
            
            input = torch.stack((estim,noisy),0)
            
            input = torch.unsqueeze(input,0).to(device)
            
            ## Run 

            mask = model(input)

            # [B, (estim,noisy), F, T, Cplx]
            mag_output = input[:, 0, :, :] * mask
            
            cplx_output = mag_output*torch.exp(phase.to(device)*1j)

            wav_output = torch.istft(cplx_output, n_fft, hop_length=None, win_length=None, window=window, center=True, normalized=False, onesided=None, length=None, return_complex=False)

            wav_output = wav_output.to('cpu')

            ## Normalize
            max_val = torch.max(torch.abs(wav_output))
            wav_output= wav_output/max_val

            ## Save
            torchaudio.save(os.path.join(args.dir_output,name_wav),src=wav_output[:,:],sample_rate=hp.audio.samplerate,bits_per_sample=16)
