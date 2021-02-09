import torch
import argparse
import torchaudio
import os
import numpy as np
from tqdm import tqdm

from model.ModelFC import ModelFC
from dataset.TestsetFC import TestsetFC

from utils.hparams import HParam

def spec2wav(complex_ri, window, length):
    audio = torch.istft(input= complex_ri, n_fft=int(1024), hop_length=int(256), win_length=int(1024), window=window, center=True, normalized=False, onesided=True, length=length)
    return audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    device = hp.gpu
    torch.cuda.set_device(device)

    batch_size = 1
    block = hp.model.FC.block
    num_epochs = 1
    num_workers = hp.train.num_workers

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    list_test= ['dt05_bus_real','dt05_caf_real','dt05_ped_real','dt05_str_real','et05_bus_real','et05_caf_real','et05_ped_real','et05_str_real','dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']
    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)
    for i in list_test :
        os.makedirs(os.path.join(output_dir,i),exist_ok=True)

    test_dataset= TestsetFC(hp.data.root+'/STFT_R',list_test,'*.npy',block=block)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = ModelFC(hp).to(device)

    print('NOTE::Loading pre-trained model : '+ args.model)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            spec_input = data["input"].to(device)

            output=[]
            for j in range(data["length"]) :
                #print(str(j) +" | "+ str(data["length"]))
                # Note : abandon block-frames in edges
                if j < block :
                    continue
                if j > data["length"] - 2*block-1 :
                    break
                #print("spec_input : " + str(spec_input.shape))

                torch_input = torch.reshape(spec_input[:,:,:,j:j+2*block+1,:],(513*3*(2*block+1),2))
                #print("torch_input : " + str(torch_input.shape))

                # for batch 1 
                torch_input = torch.unsqueeze(torch_input,0)

                output_cur = model(torch_input)
                print(output_cur[0,:,0])
                print("output_cur : " + str(output_cur.shape))

                # for dim of frame_num
                output_cur = torch.unsqueeze(output_cur,-1)
                #print("output_cur : " + str(output_cur.shape))
                if j == block : 
                    output = output_cur
                else : 
                    output = torch.cat((output,output_cur),3)
                #print("output : " + str(output.shape))

            output = output.permute(0,1,3,2)
            #print("output : " + str(output.shape))

            wav = spec2wav(output,window,int(data["length"] - 2*block -1)*hp.audio.shift)
            wav = wav.to('cpu')

            ## Normalize
            #max_var = torch.max(torch.abs(wav))
            #wav = wav/max_var

            ## save
            torchaudio.save(output_dir+'/'+str(data["dir"][0]) + '/'+str(data["name"][0])+'.wav',src = wav, sample_rate = hp.audio.samplerate)
            
            break
     