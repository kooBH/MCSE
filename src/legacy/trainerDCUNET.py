import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa

from tensorboardX import SummaryWriter

from model.DCUNET import DCUNET
from model.MDCUNET import MDCUNET
from dataset.DatasetDCUNET import DatasetDCUNET
from dataset.TestsetDCUNET import TestsetDCUNET

from utils.hparams import HParam
from utils.Loss import LossBundle
from utils.writer import MyWriter

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
    num_frame = hp.model.DCUNET.num_frame
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    best_loss = 100

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    SNR_train= hp.data.SNR
    #SNR_train= ['SNR0']

    train_dataset = DatasetDCUNET(hp.data.root+'train',SNR_train,num_frame=num_frame)
    val_dataset = DatasetDCUNET(hp.data.root+'test',SNR_train,num_frame=num_frame)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    if hp.model == 'MDCUNET' :
        print("model : MDCUNET")
        model = MDCUNET(dropout = hp.model.DCUNET.dropout,complex=hp.model.DCUNET.complex).to(device)
    
    else : 
        print("model : DCUNET")
        model = DCUNET(dropout = hp.model.DCUNET.dropout,complex=hp.model.DCUNET.complex).to(device)
    loss= LossBundle(hp,device)
    
    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if hp.loss.type == 'mSDR': 
        criterion = loss.mSDRLoss
    elif hp.loss.type == 'iSDR':
        criterion = loss.iSDRLoss
    elif hp.loss.type == 'wSDR' : 
        criterion = loss.wSDRLoss
    elif hp.loss.type == 'MSE':
        criterion = torch.nn.MSELoss()
    elif hp.loss.type == 'SDR':
        criterion = loss.SISDR
    elif hp.loss.type == 'mwMSE':
        criterion = loss.mwMSE  
    elif hp.loss.type == 'mwMSE_iSDR' : 
        criterion = loss.mwMSE_iSDR
    else : 
        raise Exception('Unknown loss function')

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

            mask_r, mask_i = model(input)

            enhance_r = input[:, 0, :, :, 0] * mask_r
            enhance_i = input[:, 0, :, :, 1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)

            if hp.loss.type == 'wSDR' : 
                noisy = batch_data["input"][:,0,:,:,:].to(device)
                loss = criterion(enhance_spec,noisy,clean).to(device)
            else :
                loss = criterion(enhance_spec,clean).to(device)

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
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

                mask_r, mask_i = model(input)

                enhance_r = input[:, 0, :, :, 0] * mask_r
                enhance_i = input[:, 0, :, :, 1] * mask_i

                enhance_r = enhance_r.unsqueeze(3)
                enhance_i = enhance_i.unsqueeze(3)
                enhance_spec = torch.cat((enhance_r,enhance_i),3)
                if hp.loss.type == 'wSDR' : 
                    noisy = batch_data["input"][:,0,:,:,:].to(device)
                    loss = criterion(enhance_spec,noisy,clean).to(device)
                else :
                    loss = criterion(enhance_spec,clean).to(device)

                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            scheduler.step(val_loss)

            writer.log_value(loss,step,'test loss['+hp.loss.type+']')

            ### Tensorboard log for specific sample
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



            # [Freq, Time, complex] 
            torch_real_noisy = torch.from_numpy(npy_real_noisy)
            torch_real_estim = torch.from_numpy(npy_real_estim)
            torch_real_noise = torch.from_numpy(npy_real_noise)

            torch_simu_noisy = torch.from_numpy(npy_simu_noisy)
            torch_simu_estim = torch.from_numpy(npy_simu_estim)
            torch_simu_noise = torch.from_numpy(npy_simu_noise)
            torch_simu_clean = torch.from_numpy(npy_simu_clean)

            data_real = torch.stack((torch_real_noisy,torch_real_estim,torch_real_noise),0)
            data_real = torch.unsqueeze(data_real,dim=0).to(device)
            data_simu = torch.stack((torch_simu_noisy,torch_simu_estim,torch_simu_noise),0)
            data_simu = torch.unsqueeze(data_simu,dim=0).to(device)

            ## sample inference

            # [n_batch,n_channel,n_freq,n_frame,cplx]
            # real
            mask_r,mask_i = model(data_real)
            if hp.model.DCUNET.input =='estim':
                enhance_r = data_real[:, 1, :, :, 0] * mask_r
                enhance_i = data_real[:, 1, :, :, 1] * mask_i
            # default noisy
            else :
                enhance_r = data_real[:, 0, :, :, 0] * mask_r
                enhance_i = data_real[:, 0, :, :, 1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)

            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            enhance_real = torch.squeeze(enhance_spec)

            # simu
            mask_r,mask_i = model(data_simu)
            if hp.model.DCUNET.input =='estim':
                enhance_r = data_simu[:, 1, :, :, 0] * mask_r
                enhance_i = data_simu[:, 1, :, :, 1] * mask_i
            # default noisy
            else :
                enhance_r = data_simu[:, 0, :, :, 0] * mask_r
                enhance_i = data_simu[:, 0, :, :, 1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)

            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            enhance_simu = torch.squeeze(enhance_spec)

            npy_real_output = enhance_real.cpu().detach().numpy()
            npy_simu_output = enhance_simu.cpu().detach().numpy()

            audio_real_output = librosa.istft(npy_real_output[:,:,0]+npy_real_output[:,:,1]*1j)
            audio_real_noisy= librosa.istft(npy_real_noisy[:,:,0]+npy_real_noisy[:,:,1]*1j)
            audio_real_estim= librosa.istft(npy_real_estim[:,:,0]+npy_real_estim[:,:,1]*1j)

            writer.log_audio(audio_real_output, 'real_output',step)
            writer.log_audio(audio_real_noisy, 'real_noisy', step)
            writer.log_audio(audio_real_estim, 'real_estim', step)

            writer.log_spec(enhance_simu,'simu_output',step)
            writer.log_spec(torch_simu_clean,'simu_clean',step)
            writer.log_spec(torch_simu_noisy,'simu_noisy',step)
            writer.log_spec(torch_simu_estim,'simu_estim',step)
            ### Save model state

            #torch.save(model.state_dict(), str(modelsave_path)+ '/model_step_'+str(step)+'_loss_'+loss+'.pt')
            torch.save(model.state_dict(), str(modelsave_path)+ '/model_step_'+str(step)+'.pt')

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss
