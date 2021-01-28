import torch
import torch.nn as nn


class ModelFC(nn.Module):
    def __init__(self,hp):
        super().__init__()

        self.block = hp.model.FC.block
        self.block_len = 2*self.block+1
        self.hfft = int(hp.audio.frame/2 + 1)
        self.input_size = self.hfft*self.block_len*3
        print('ModelFC[input_size] : '+ str(self.input_size))

        self.model_real = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024,1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024,self.hfft),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(negative_slope=0.01)
        )
        self.model_complex = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024,1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024,self.hfft),
            torch.nn.ReLU()
            #torch.nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self,x):
        # [B, input_size, 2 ]
        output_real = self.model_real(x[:,:,0])
        output_complex = self.model_complex(x[:,:,1])

        # [B, hfft , 2]
        return torch.stack((output_real,output_complex),-1)