import sys
sys.path.append("../src/")
import os,glob

import torch
import torchaudio
import argparse
from model.DCUNET import DCUNET
from utils.hparams import HParam

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-i','--dir_input',type=str,required=True)
    parser.add_argument('-o','--dir_output',type=str,required=True)
    parser.add_argument('-d','--device',type=str,default='cuda:0')
    args = parser.parse_args()

    ## Parameters 
    hp = HParam(args.config)

    device = args.device
    torch.cuda.set_device(device)

    num_epochs = 1
    batch_size = 1
    test_model = args.model
    win_len = hp.audio.frame
    length = 16*15

    dir_output = args.dir_output
    dir_input = args.dir_input
    os.makedirs(dir_output,exist_ok=True)

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    ## Load data
    list_data = [x for x in glob.glob(os.path.join(dir_input,'noisy','*.pt'))]
    list_name = [x.split('/')[-1] for x in list_data ]

    ## Model
    model = DCUNET(complex=hp.model.DCUNET.complex).to(device)
    model.load_state_dict(torch.load(test_model,map_location=device))
    model.eval()
    print('NOTE::Loading pre-trained model : '+ args.model)

    with torch.no_grad():
        for target_name in tqdm(list_name) : 
            target_id = target_name.split('.')[0]
            # noisy
            noisy = torch.load(dir_input + '/noisy/' + target_name)
            # estim
            estim = torch.load(dir_input + '/estim/' + target_name)
            # noise
            noise = torch.load(dir_input + '/noise/' + target_name)
            # stack & unsqueeze
            input = torch.stack((noisy,estim,noise),0)
            input = torch.unsqueeze(input,0)
            input  = input.to(device)

            # num_frame must be multiple of 16
            input = input[:,:,:, :length , :]


            mask_r,mask_i = model(input)

            enhance_r = input[:,0,:,:,0]*mask_r
            enhance_i = input[:,0,:,:,1]*mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)

            enhance_wav = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window, center = True, normalized=False,onesided=True,length=length*hp.audio.shift)

            enhance_wav  = enhance_wav.to('cpu')
            enhance_spec = enhance_spec.to('cpu')

            torch.save(enhance_spec, dir_output+'/'+target_name)
            torchaudio.save(dir_output+'/'+target_id+'.wav',src=enhance_wav,sample_rate=hp.audio.samplerate,bits_per_sample=16)
