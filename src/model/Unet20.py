# Tensorflow code -> Pytorch Code 
# of vblez/Speech-enhancement
# https://github.com/vbelz/Speech-enhancement/blob/master/model_unet.py
from shutil import ExecError
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from  utils.TCN import TCN
from torch import Tensor

## Uformer Modules
## https://arxiv.org/pdf/2111.06015.pdf

class EncoderDecoderAttention(nn.Module):
    def __init__(self,channels) -> None :
        super(EncoderDecoderAttention,self).__init__()
        
        # Encoder Kernel  
        self.w_e = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        # Decoder Kernel  
        self.w_d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        # Attention Kernel  
        self.w_a = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2,3),stride=1, padding="same", dilation=1)
        
        self.sigma = nn.Sigmoid()
        
    def forward(self, d:Tensor, e:Tensor) -> Tensor:
        ## (10) extracting high dimensional feature
        g = self.sigma(self.w_e(e) + self.w_d(d))
        
        ## (11) multipling attention masking
        d_hat = torch.mul(self.sigma(self.w_a(g)),d)
        return d_hat


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros",dropout=0,activation="LeakyReLU"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        conv = nn.Conv2d
        bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.acti = None
        if activation == "LeakyReLU":
            self.acti = nn.LeakyReLU(inplace=True)
        elif activation == "SiLU":
            self.acti = nn.SiLU(inplace=True)
        elif activation == 'Softplus':
            self.acti = nn.Softplus()
        elif activation == 'PReLU':
            self.acti = nn.PReLU()
        elif activation == 'ReLU':
            self.acti = nn.ReLU()
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0),activation="LeakyReLU"):
        super().__init__()
       
        tconv = nn.ConvTranspose2d
        bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)
        self.acti = None
        if activation == "LeakyReLU":
            self.acti = nn.LeakyReLU(inplace=True)
        elif activation == "SiLU":
            self.acti = nn.SiLU(inplace=True)
        elif activation == 'Softplus':
            self.acti = nn.Softplus()
        elif activation == 'PReLU':
            self.acti = nn.PReLU()
        elif activation == 'ReLU':
            self.acti = nn.ReLU()
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.acti(x)
        return x

class ResPath(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.resconv1(x)
        z1 = x1+y1
        x2 = self.conv2(z1)
        y2 = self.resconv2(z1)
        z2 = x2+y2
        x3 = self.conv3(z2)
        y3 = self.resconv3(z2)
        z3 = x3+y3
        x4 = self.conv4(z3)
        y4 = self.resconv4(z3)
        z4 = x4+y4
        return z4

class OberservedAddition(nn.Module):
# Inspired by 
# Iwamoto, Kazuma, et al. "How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR." arXiv preprint arXiv:2201.06685 (2022).
# https://arxiv.org/abs/2201.06685

    def __init__(self, method="none", dim=-1,factor=0.3):
        super().__init__()
        self.method = method
        self.dim = dim
        
        if method == "conv":
            self.OA = nn.Conv2d(2,1,1) 
        elif method == "linear":
            if dim == -1 :
                raise Exception("ERROR::ObversedAddtion requires 'dim' for linear layer")
            self.OA = nn.Linear(2*dim,dim)
        elif method == 'const':
            # self.factor = nn.Parameter(torch.ones(1))
            self.factor = nn.Parameter(torch.tensor(factor))
        elif method == 'fixed' : 
            self.factor = factor
        else :
            self.OA = nn.Identity()

    def forward(self, x, o):
        # x : [B C F T]
        # o : [B C F T]
        if self.method == 'conv':
            # concate on channel
            return self.OA(torch.cat((x,o),dim=1))
        elif self.method == 'linear' :
            # concate on feature
            return self.OA(torch.cat((x,o),dim=2))
        elif self.method == 'const' or self.metohd == 'fixed':
            return x + self.factor * o
        else :
            return self.OA(x)

class Unet20(nn.Module):
    def __init__(self, hp,
                 device='cuda:0',
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros",
                 use_EDA=False, # Enable Encoder Decoder Attention
                 OA_method = "none",
                 OA_dim = 257,
                 OA_factor=0.3
                 ):
        super().__init__()

        self.hp = hp
        self.device = device
        
        input_channels = hp.model.UNET.channels
        dropout = hp.model.UNET.dropout
        activation = hp.model.UNET.activation
        mask_activation = hp.model.UNET.mask_activation
        
        self.nhfft = hp.audio.frame/2 + 1

        # WIP : respath, EDA 맞추기 
        self.use_respath = hp.model.UNET.use_respath

       
        model_complexity = int(model_complexity // 1.414)

        self.enc_channels = [input_channels,
                                model_complexity,
                                model_complexity,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                128]

        self.enc_kernel_sizes = [   (7, 1),
                                    (1, 7),
                                    (7, 5),
                                    (7, 5),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3)]

        self.enc_strides = [(1, 1),
                            (1, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1)]

        self.enc_paddings = [(3, 0),
                                (0, 3),
                                (3, 2),
                                (3, 2),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),]
        self.dec_channels = [0,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2]

        self.dec_kernel_sizes = [(5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3), 
                                    (7, 5), 
                                    (7, 5), 
                                    (1, 7),
                                    (7, 1)]

        self.dec_strides = [(2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (1, 1),
                            (1, 1)]

        self.dec_paddings = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (3, 2),
                                (3, 2),
                                (0, 3),
                                (3, 0)]
        self.dec_output_paddings = [(0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,0)]

        self.encoders = []
        self.model_length = model_depth // 2


        if use_EDA :
            self.dec_channels *= 2
            
            self.EDA = []
            for i in range(self.model_length):
                module = EncoderDecoderAttention(self.enc_channels[self.model_length-i])
                self.add_module("EDA{}".format(i),module)
                self.EDA.append(module)

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i],  padding_mode=padding_mode,dropout=dropout,activation=activation)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], output_padding=self.dec_output_paddings[i],activation=activation)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if self.use_respath : 
            self.respaths = [] 
            for i in range(self.model_length) :
                module = ResPath(self.enc_channels[i])
                self.add_module("respath{}".format(i),module)
                self.respaths.append(module)

        ## Bottlenect
        self.bottleneck = hp.model.UNET.bottleneck
        if hp.model.UNET.bottleneck == 'None' :
            self.bottleneck = nn.Identity()
        elif hp.model.UNET.bottleneck == 'GRU':
            # case for F = 513 
            self.bottleneck = nn.GRU(input_size  = 128*3,hidden_size = 64*3,num_layers = 2,bias=True,batch_first=True,bidirectional=True,dropout =hp.model.UNET.bottleneck_dropout)
        elif hp.model.UNET.bottleneck == 'LSTM':
            # case for F = 513 
            self.bottleneck = nn.LSTM(input_size  = 128*3,hidden_size = 64*3,num_layers = 2,bias=True,batch_first=True,bidirectional=True,dropout=hp.model.UNET.bottleneck_dropout)
        elif hp.model.UNET.bottleneck == 'LSTM2':
            # case for F = 513 
            self.bottleneck = nn.LSTM(input_size  = 128*3,hidden_size = 128*3,num_layers = 2,bias=True,batch_first=True,bidirectional=True,dropout=hp.model.UNET.bottleneck_dropout)
        elif hp.model.UNET.bottleneck == 'LSTM3':
            # case for F = 513 
            self.bottleneck = nn.LSTM(input_size  = 128*3,hidden_size = 64*3,num_layers = 3,bias=True,batch_first=True,bidirectional=True,dropout=hp.model.UNET.bottleneck_dropout)
        elif hp.model.UNET.bottleneck == 'TCN':
            self.bottleneck = TCN(c_in=128*3, c_out=[128*3,128*3])
        else :
            raise Exception("ERROR:no bottleneck")

        
        linear = nn.Conv2d(self.dec_channels[-1], 1, 1)
        self.mask_acti = None
        if mask_activation == 'Sigmoid' : 
            self.mask_acti = nn.Sigmoid()
        elif mask_activation == 'ReLU' : 
            self.mask_acti = nn.ReLU()
        elif mask_activation == 'Softplus':
            # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus
            self.mask_acti = nn.Softplus()
        elif mask_activation == "SiLU":
            self.mask_acti = nn.SiLU()
        elif mask_activation == 'none':
            self.mask_acti = nn.Identity()
        else :
            raise Exception('ERROR:Unknown activation : ' + str(activation))

        self.add_module("linear", linear)
        self.padding_mode = padding_mode
        
        OA = OberservedAddition(method = OA_method,dim=OA_dim,factor=OA_factor)
        self.add_module('OA',OA)

    def forward(self, x):        
        # ipnut : [ Batch Channel Freq Time]

        # Encoder 
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            if self.use_respath : 
                x_skip.append(self.respaths[i](x))
            else :
                x_skip.append(x)
            x = encoder(x)
            #print("x{}".format(i), x.shape)
        # x_skip : x0=input x1 ... x9

        #print("fully encoded ",x.shape)
        if self.hp.model.UNET.bottleneck == 'GRU' or self.hp.model.UNET.bottleneck == 'LSTM':
            # [B, C, F, T]
            B, C, F, T = x.shape
            x = torch.permute(x,(0,3,1,2))
            x = torch.reshape(x,(B,T,C*F))
            p,_ = self.bottleneck(x)
            p = torch.reshape(p,(B,T,C,F))
            p = torch.permute(p,(0,2,3,1))
        elif self.hp.model.UNET.bottleneck == 'TCN':
            B, C, F, T = x.shape
            x = torch.reshape(x,(B,C*F,T))
            p = self.bottleneck(x)
            p = torch.reshape(p,(B,C,F,T))
        else :
            p = x
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {x_skip[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")

            # last layer of Decorders
            if i == self.model_length - 1:
                break
            p = torch.cat([p, x_skip[self.model_length - 1 - i]], dim=1)

        #print('p : ' +str(p.shape))
        mask = self.linear(p)
        mask = self.mask_acti(mask)
        
        # Observation Addition
        mask_oberserved = torch.ones(mask.shape).to(self.device)
        mask = self.OA(mask,mask_oberserved)
        return mask[:,0,:,:]
    
if __name__ == '__main__':
    path_data_sample = '/home/data/kbh/MCSE/CGMM_RLS_MPDR/train/SNR-5/noisy/011_011C0201.pt'
    input = torch.load(path_data_sample)
    input = input[:,:256,:]
    input = torch.sqrt(input[:,:,0]**2 + input[:,:,1]**2) 
    # batch
    input = torch.unsqueeze(input,dim=0)
    # channel
    input = torch.unsqueeze(input,dim=0)
    print(input.shape)

    model = Unet20()

    output = model(input)
    print(output.shape)
