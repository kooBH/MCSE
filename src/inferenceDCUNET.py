import torch
import argparse
import numpy as np
import torchaudio
import os

from model.ModelDCUNET import ModelDCUNET
from dataset.TestsetDCUNET import TestsetDCUNET
from tqdm import tqdm
from utils.hparams import HParam

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()


    ## Parameters 
    hp = HParam(args.config)
    print('NOTE::Loading configuration :: ' + args.config)

    device = hp.gpu
    torch.cuda.set_device(device)

    num_epochs = 1
    batch_size = 1
    test_model = args.model
    win_len = hp.audio.frame

    window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
 

    ## dirs 
    list_test= ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)
    for i in list_test :
        os.makedirs(os.path.join(output_dir,i),exist_ok=True)

    ## Dataset
    test_dataset = TestsetDCUNET(hp.data.root+'/STFT_R',list_test,'*.npy',num_frame=hp.model.DCUNET.num_frame)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,num_workers=1)

    ## Model
    model = ModelDCUNET(input_channels=3).to(device)
    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    print('NOTE::Loading pre-trained model : ' + test_model)

    ## Inference
    with torch.no_grad():
        for i, (data, length, data_dir, data_name) in enumerate(tqdm(test_loader)):
            spec_input = data.to(device)
            mask_r,mask_i = model(spec_input)

            # [B, (noisy,noise,clean), F, T, Cplx]
            enhance_r = spec_input[:,0,:,:,0] * mask_r
            enhance_i = spec_input[:,0,:,:,1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)

            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window, center = True, normalized=False,onesided=True,length=int(length)*hp.audio.shift)


            audio_me_pe=audio_me_pe.to('cpu')

            ## Normalize
            max_val = torch.max(torch.abs(audio_me_pe))
            audio_me_pe = audio_me_pe/max_val

            ## Save
            torchaudio.save(output_dir+'/'+str(data_dir[0])+'/'+str(data_name[0])+'.wav',src=audio_me_pe[:,:],sample_rate=hp.audio.samplerate)

