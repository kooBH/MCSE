import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from model.ModelDCUNET import ModelDCUNET
from dataset.DatasetDCUNET import DatasetDCUNET

from utils.hparams import HParam
from utils.wSDRLoss import wSDRLoss
from utils.writer import MyWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=int,required=False,default=0)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_frame = hp.model.DCUNET.num_frame
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    best_loss = 10

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    list_train= ['tr05_bus_simu','tr05_caf_simu','tr05_ped_simu','tr05_str_simu']
    list_test= ['dt05_bus_simu','dt05_caf_simu','dt05_ped_simu','dt05_str_simu','et05_bus_simu','et05_caf_simu','et05_ped_simu','et05_str_simu']

    train_dataset = DatasetDCUNET(hp.data.root+'/STFT_R',hp.data.root+'/WAV',list_train,'*.npy',num_frame=num_frame,channels = 2)
    val_dataset   = DatasetDCUNET(hp.data.root+'/STFT_R',hp.data.root+'/WAV',list_test,'*.npy',num_frame=num_frame, channels = 2)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = ModelDCUNET(input_channels=2).to(device)
    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    criterion = wSDRLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

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
        raise TypeError("Unsupported sceduler type")

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1

            spec_input = batch_data["input"].to(device)
            wav_noisy= batch_data["noisy"].to(device)
            wav_clean= batch_data["clean"].to(device)
            
            mask_r, mask_i = model(spec_input)

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
            audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window, center = True, normalized=False,onesided=True,length=num_frame*hp.audio.shift)


            loss = criterion(wav_noisy,wav_clean,audio_me_pe,eps=1e-8).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_training(loss,step)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pth')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            val_loss =0.
            for j, (batch_data) in enumerate(val_loader):
              
                spec_input = batch_data["input"].to(device)
                wav_noisy  = batch_data["noisy"].to(device)
                wav_clean  = batch_data["clean"].to(device)
            
                mask_r, mask_i = model(spec_input)

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
                audio_me_pe = torch.istft(enhance_spec,n_fft=hp.audio.frame, hop_length = hp.audio.shift, window=window, center = True, normalized=False,onesided=True,length=num_frame*hp.audio.shift).to(device)                
                loss = criterion(wav_noisy,wav_clean,audio_me_pe,eps=1e-8).to(device)
                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            scheduler.step(val_loss)

            input_audio = wav_noisy[0].cpu().numpy()
            target_audio= wav_clean[0].cpu().numpy()
            audio_me_pe= audio_me_pe[0].cpu().numpy()

            writer.log_evaluation(val_loss,step,
                                  input_audio,target_audio,audio_me_pe)

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss

