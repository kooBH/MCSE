import torch
import torch.nn as nn

"""
2021.01.28 
    add batch-norm
2021.01.29
    add dropout

"""

class ModelFC(nn.Module):
    def __init__(self,hp):
        super().__init__()

        self.block = hp.model.FC.block
        self.block_len = 2*self.block+1
        self.hfft = int(hp.audio.frame/2 + 1)
        self.input_size = self.hfft*self.block_len*3

        dropout = hp.model.FC.dropout

#        print('ModelFC[input_size] : '+ str(self.input_size))

        if hp.model.FC.version == 2 : 
            self.model_real = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.ReLU()
            )
            self.model_complex = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.ReLU()
            )
        elif hp.model.FC.version == 3 :
            self.model_real = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.ReLU()
            )
            self.model_complex = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.ReLU()
            )
        elif hp.model.FC.version == 4:
            self.model_real = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2048,2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2048,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.LeakyReLU(negative_slope=0.01)
            )
            self.model_complex = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2048,2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(2048,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.LeakyReLU(negative_slope=0.01)
            )
        elif hp.model.FC.version == 5 :
             self.model_real = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.tanh()
            )
            self.model_complex = torch.nn.Sequential(
                torch.nn.Linear(self.input_size,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,1024),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1024,self.hfft),
                torch.nn.tanh()
            )

        else :
            raise Exception("Unknown model version")

    def forward(self,x):
        # [B, input_size, 2 ]
        output_real = self.model_real(x[:,:,0])
        output_complex = self.model_complex(x[:,:,1])

        # [B, hfft , 2]
        return torch.stack((output_real,output_complex),-1)