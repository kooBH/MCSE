import torch
import argparse
import numpy as np
import torchaudio
import os

from model.DCUNET import DCUNET
from dataset.TestsetDCUNET import TestsetDCUNET
from tqdm import tqdm
from utils.hparams import HParam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-o','--output_dir',type=str,required=True)
    parser.add_argument('-t','--type',type=str,default='respective')
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
    post_filter = hp.model.DCUNET.post_filter

    window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
 
    ## dirs 
    list_test= ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real','et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real','dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)
    for i in list_test :
        os.makedirs(os.path.join(output_dir,i),exist_ok=True)

    ## Dataset
    test_dataset = TestsetDCUNET(hp.data.test_root+'/STFT_R',list_test,'*.npy',num_frame=hp.model.DCUNET.num_frame)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=1)

    ## Model
    model = DCUNET(input_channels=3).to(device)
    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    print('NOTE::Loading pre-trained model : ' + test_model)

    ## Inference
    with torch.no_grad():
        for i, (data, length, data_dir, data_name) in enumerate(tqdm(test_loader)):
            spec_input = data.to(device)
            mask_r,mask_i = model(spec_input)

            # [B, (noisy,estim,noise), F, T, Cplx]
            if hp.model.DCUNET.input =='estim':
                enhance_r = spec_input[:, 1, :, :, 0] * mask_r
                enhance_i = spec_input[:, 1, :, :, 1] * mask_i
            # default noisy
            else :
                enhance_r = spec_input[:, 0, :, :, 0] * mask_r
                enhance_i = spec_input[:, 0, :, :, 1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)

            enhance_spec = torch.cat((enhance_r,enhance_i),3)

            if args.type == 'respective' : 
                audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window, center = True, normalized=False,onesided=True,length=int(length)*hp.audio.shift)
            elif args.type == 'magnitude':
                enhance_mag,enhance_phase = torchaudio.functional.magphase(enhance_spec) 

                noisy_spec = torch.cat((spec_input[:,0,:,:,0].unsqueeze(3),spec_input[:,0,:,:,1].unsqueeze(3)),3)
                noisy_mag,noisy_phase = torchaudio.functional.magphase(noisy_spec)

                output_spec = enhance_mag * torch.exp(1j*noisy_phase)

                output_spec = torch.view_as_real(output_spec)
                audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window,center =True, normalized=False,onesided=True,length=int(length)*hp.audio.shift)
            elif args.type == 'magnitude_with_estim_phase' :
                enhance_mag,enhance_phase = torchaudio.functional.magphase(enhance_spec) 

                estim_spec = torch.cat((spec_input[:,0,:,:,0].unsqueeze(3),spec_input[:,0,:,:,1].unsqueeze(3)),3)
                estim_mag,estim_phase = torchaudio.functional.magphase(estim_spec)

                output_spec = enhance_mag * torch.exp(1j*estim_phase)

                output_spec = torch.view_as_real(output_spec)
                audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window,center =True, normalized=False,onesided=True,length=int(length)*hp.audio.shift)
            else :
                raise TypeError('Unknown mask type')

            audio_me_pe=audio_me_pe.to('cpu')

            ## Normalize
            max_val = torch.max(torch.abs(audio_me_pe))
            audio_me_pe = audio_me_pe/max_val

            ## Save
            torchaudio.save(output_dir+'/'+str(data_dir[0])+'/'+str(data_name[0])+'.wav',src=audio_me_pe[:,:],sample_rate=hp.audio.samplerate,bits_per_sample=16)
