import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa

from tensorboardX import SummaryWriter

from model.Unet20 import Unet20
from dataset.DatasetUNET import DatasetUNET

from utils.hparams import HParam
from utils.writer import MyWriter
from utils.Loss import LossBundle

def logging(hp,model,writer,step):
    npy_real_noisy = np.load(hp.data.test_root + '/STFT_R/noisy/'+hp.data.sample_real + '.npy')
    npy_real_estim = np.load(hp.data.test_root + '/STFT_R/estim/'+hp.data.sample_real + '.npy')
    npy_real_noise = np.load(hp.data.test_root + '/STFT_R/noise/'+hp.data.sample_real + '.npy')

    npy_simu_noisy = np.load(hp.data.test_root + '/STFT_R/noisy/'+hp.data.sample_simu + '.npy')
    npy_simu_estim = np.load(hp.data.test_root + '/STFT_R/estim/'+hp.data.sample_simu + '.npy')
    npy_simu_noise = np.load(hp.data.test_root + '/STFT_R/noise/'+hp.data.sample_simu + '.npy')
    npy_simu_clean = np.load(hp.data.test_root + '/STFT_R/clean/'+hp.data.sample_simu + '.npy')

    ## length must be multiply of 16
    # simu
    length_simu = np.size(npy_simu_noisy,1)
    target_length_simu = int(16*np.floor(length_simu/16)+16)
    if length_simu < target_length_simu : 
        need = target_length_simu - length_simu
        npy_simu_noisy =  np.pad(npy_simu_noisy,((0,0),(0,need),(0,0)),'constant',constant_values=0)
        npy_simu_noise =  np.pad(npy_simu_noise,((0,0),(0,need),(0,0)),'constant',constant_values=0)
        npy_simu_estim =  np.pad(npy_simu_estim,((0,0),(0,need),(0,0)),'constant',constant_values=0)
    # real
    length_real = np.size(npy_real_noisy,1)
    target_length_real = int(16*np.floor(length_real/16)+16)
    if length_real < target_length_real : 
        need = target_length_real - length_real
        npy_real_noisy =  np.pad(npy_real_noisy,((0,0),(0,need),(0,0)),'constant',constant_values=0)
        npy_real_noise =  np.pad(npy_real_noise,((0,0),(0,need),(0,0)),'constant',constant_values=0)
        npy_real_estim =  np.pad(npy_real_estim,((0,0),(0,need),(0,0)),'constant',constant_values=0)

    # numpy -> torch
    torch_real_noisy = torch.from_numpy(npy_real_noisy)
    torch_real_estim = torch.from_numpy(npy_real_estim)
    torch_real_noise = torch.from_numpy(npy_real_noise)

    torch_simu_noisy = torch.from_numpy(npy_simu_noisy)
    torch_simu_estim = torch.from_numpy(npy_simu_estim)
    torch_simu_noise = torch.from_numpy(npy_simu_noise)
    torch_simu_clean = torch.from_numpy(npy_simu_clean)

    # [1, C, F, T]
    data_real = torch.stack((torch_real_noisy,torch_real_estim,torch_real_noise),0)
    data_real = torch.sqrt(data_real[:,:,:,0]**2 + data_real[:,:,:,1]**2)
    data_real = torch.unsqueeze(data_real,dim=0).to(device)
    data_simu = torch.stack((torch_simu_noisy,torch_simu_estim,torch_simu_noise),0)
    data_simu = torch.sqrt(data_simu[:,:,:,0]**2 + data_simu[:,:,:,1]**2)
    data_simu = torch.unsqueeze(data_simu,dim=0).to(device)

    ## sample inference
    # [n_batch,n_channel,n_freq,n_frame]
    # real

    #       B - 
    #

    if hp.model.UNET.method == 'masking':  
        mask = model(data_real[:, :hp.model.UNET.channels, :, :])
        output_mag_real = data_real[0,0,:,:] * mask

        mask = model(data_simu[:, :hp.model.UNET.channels, :, :])
        output_mag_simu = data_simu[0,0,:,:]*mask
    else :
        output_mag_real = model(data_real[:, :hp.model.UNET.channels, :, :])
        output_mag_simu = model(data_simu[:, :hp.model.UNET.channels, :, :])
    output_mag_real = torch.squeeze(output_mag_real,0)
    output_mag_simu = torch.squeeze(output_mag_simu,0)

    # torch -> numpy 
    npy_mag_real = output_mag_real.cpu().detach().numpy()
    npy_mag_simu = output_mag_simu.cpu().detach().numpy()

    ## phase of real
    phase_real = None
    cplx_real = None
    if hp.model.UNET.input == 'noisy' :
        phase_real = np.angle(npy_real_noisy[:,:,0]+npy_real_noisy[:,:,1]*1j)
        cplx_real = npy_mag_real[:,:]*np.exp(phase_real*1j)
    else:
        phase_real = np.angle(npy_real_estim[:,:,0]+npy_real_estim[:,:,1]*1j)
        cplx_real = npy_mag_real[:,:]*np.exp(phase_real*1j)

    ## mag phase to cplx
    # mag * exp(1j*ang)

    audio_real_output = librosa.istft(cplx_real)
    audio_real_noisy= librosa.istft(npy_real_noisy[:,:,0]+npy_real_noisy[:,:,1]*1j)
    audio_real_estim= librosa.istft(npy_real_estim[:,:,0]+npy_real_estim[:,:,1]*1j)

    ## Normalization
    audio_real_output = audio_real_output/np.max(audio_real_output)

    writer.log_audio(audio_real_output, 'real_output',step)
    writer.log_audio(audio_real_noisy, 'real_noisy', step)
    writer.log_audio(audio_real_estim, 'real_estim', step)

    writer.log_spec(output_mag_simu,'simu_output',step)
    writer.log_spec(torch_simu_clean,'simu_clean',step)
    writer.log_spec(torch_simu_noisy,'simu_noisy',step)
    writer.log_spec(torch_simu_estim,'simu_estim',step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default='cuda:0')
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    torch.cuda.set_device(device)
    print(device)

    batch_size = hp.train.batch_size
    num_frame = hp.model.UNET.num_frame
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    best_loss = 1e3

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    SNR_train= hp.data.SNR
    #SNR_train= ['SNR0']

    train_dataset = DatasetUNET(hp,is_train=True)
    val_dataset = DatasetUNET(hp,is_train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # Set Model
    if hp.model.UNET.type == 'Unet20' : 
        print("model : Unet20")
        model = Unet20(hp).to(device)
    elif hp.model.UNET.type == 'TRU':
        print("model : UNET Tiny Recurrent Unet")
        model = UNET().to(device)
    else :
        raise Exception('No model such as '+ str(hp.model.UNET.type))

    # load pre-trained
    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    loss = LossBundle(hp,device)
    
    ## Loss
    inSTFT=True
    to_be_wav = False
    if hp.loss.type == 'mSDR':
        criterion = loss.mSDRLoss
        to_be_wav = True
    elif hp.loss.type == 'wSDR':
        criterion = loss.wSDRLoss
        to_be_wav = True
    elif hp.loss.type == 'iSDR':
        criterion = loss.iSDRLoss
        to_be_wav = True
    elif hp.loss.type == 'wMSE':
        criterion = loss.wMSE
    elif hp.loss.type == 'mwMSE':
        criterion = loss.mwMSE
    elif hp.loss.type == 'MSE':
        criterion = nn.MSELoss
    else :
        raise Exception('Unknown loss function : ' + str(hp.loss.type))

    # Optimizer
    if hp.optimizer.type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.optimizer.Adam.lr)
    elif hp.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=hp.optimizer.AdamW.lr, 
            betas=(hp.optimizer.AdamW.betas[0],hp.optimizer.AdamW.betas[1]),
            weight_decay = hp.optimizer.AdamW.weight_decay,
            amsgrad = hp.optimizer.AdamW.amsgrad
            )
    else :
        raise Exception('Unknown optimizer : ' + str(hp.train.optimizer))

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    else :
        raise TypeError("Unsupported scheduler type")

    step = args.step

    # to detect NAN
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1

            input = batch_data["input"].to(device)
            clean= batch_data["clean"].to(device)


            if hp.model.UNET.method == 'masking': 
                mask = model(input[:,:hp.model.UNET.channels,:,:])
            # [n_batch, n_channel, n_freq, n_time]
                output = input[:, 0, :, :] * mask
            else :
                output = model(input[:,:hp.model.UNET.channels,:,:])

            if to_be_wav :
                output = output*torch.exp(batch_data['phase'][:,0,:,:].to(device)*1j)
                clean  = clean *torch.exp(batch_data['phase'][:,1,:,:].to(device)*1j)

            loss = criterion(output,clean,inSTFT=inSTFT).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('TRAIN::{} - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(args.version_name,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss' +'['+hp.loss.type+']')

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            val_loss =0.
            for j, (batch_data) in enumerate(val_loader):
                input = batch_data["input"].to(device)
                clean  = batch_data["clean"].to(device)

                if hp.model.UNET.method == 'masking': 
                    mask = model(input[:,:hp.model.UNET.channels,:,:])
                    output = input[:, 0, :, :] * mask
                else : 
                    output  = model(input[:,:hp.model.UNET.channels,:,:])

                if to_be_wav :
                    output = output*torch.exp(batch_data['phase'][:,0,:,:].to(device)*1j)
                    clean  = clean *torch.exp(batch_data['phase'][:,1,:,:].to(device)*1j)

                loss = criterion(output,clean,inSTFT=inSTFT).to(device)
                
                print('TEST::{} - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(args.version_name,epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            if hp.scheduler.type == 'oneCycle':
                scheduler.step()
            else :
                scheduler.step(val_loss)

            writer.log_value(val_loss,step,'test loss['+hp.loss.type+']')

            ### Tensorboard log for specific sample
            logging(hp,model,writer,step) 

            ### Save model state

            #torch.save(model.state_dict(), str(modelsave_path)+ '/model_step_'+str(step)+'_loss_'+loss+'.pt')
            #torch.save(model.state_dict(), str(modelsave_path)+ '/model_step_'+str(step)+'.pt')

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss
