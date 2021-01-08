# generic
import os, glob
# process
import numpy as np
import torch
import torchaudio
import librosa
import scipy
import scipy.io
import soundfile

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

fft_size = 1024
mel_band = 13

input_root = '/home/data/kbh/3-channel-dnn/STFT/'
output_root= '/home/data/kbh/3-channel-dnn/LMPSC'+str(mel_band)+'/'

target_list = [x for x in glob.glob(os.path.join(input_root,'*','*','*.npy')) if not os.path.isdir(x)]

def  generate(idx):
    target_path = target_list[idx]
    target = np.load(target_path)
    target =  librosa.feature.melspectrogram( S=target, n_fft=fft_size, hop_length=int(fft_size/4),win_length=None, window='hann', center=False, pad_mode='reflect', power=2.0, n_mels=mel_band)

    # modify path
    # STFT/clean/dt05_str_simu/A.npy
    # => LMPSC/clean/dt05_str_simu/A.npy
    output_path = target_path.split('/')
    output_path = output_root + '/'+ output_path[-3]+'/'+output_path[-2]+'/'+output_path[-1]

    np.save(output_path,target)

if __name__=='__main__' : 
    cpu_num = cpu_count()

    list_category = ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu','tr05_bus_simu','tr05_caf_simu','tr05_ped_simu','tr05_str_simu']
    list_dir = ['noise','noisy','clean','estim']
    try:
       os.mkdir(output_root)
    except FileExistsError:
       pass
    for i in list_dir : 
        try:
            os.mkdir(output_root + i)
        except FileExistsError:
            pass
        for j in list_category :
            try:
                os.mkdir(output_root + i+'/'+j)
            except FileExistsError:
                pass

    arr = list(range(len(target_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generate, arr), total=len(arr),ascii=True,desc='LMPSC'+str(mel_band)))






