import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from model.DCUNET import DCUNET
from dataset.DatasetDCUNET import DatasetDCUNET
from dataset.TestsetDCUNET import TestsetDCUNET

from utils.hparams import HParam
from utils.SISDR import SDR
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

    SNR_train= ['SNR-7','SNR-5','SNR0','SNR5','SNR7','SNR10']
    #SNR_train= ['SNR0']

    raw_dataset = DatasetDCUNET(hp.data.root,SNR_train,num_frame=num_frame)
    len_dataset = len(raw_dataset)

    # 99 : 1 = train : test
    len_trainset = int(len_dataset * 0.99)
    len_testset = len_dataset - len_trainset

    train_dataset, val_dataset = torch.utils.data.random_split(raw_dataset,[len_trainset,len_testset],generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = DCUNET().to(device)
    loss_class = SDR(device)
    
    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if hp.loss.type == 'SDR' : 
        criterion = loss_class.SDRLoss
    elif hp.loss.type == 'mSDR': 
        criterion = loss_class.mSDRLoss
    else : 
        raise Exception('Unknown loss function')
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

            input = batch_data["input"].to(device)
            clean= batch_data["clean"].to(device)

            mask_r, mask_i = model(input)

            enhance_r = input[:, 0, :, :, 0] * mask_r
            enhance_i = input[:, 0, :, :, 1] * mask_i

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)

            loss = criterion(enhance_spec,clean).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss')

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
                loss = criterion(enhance_spec,clean).to(device)

                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            scheduler.step(val_loss)

            writer.log_value(loss,step,'test loss')

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss

