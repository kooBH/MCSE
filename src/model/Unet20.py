# Tensorflow code -> Pytorch Code 
# of vblez/Speech-enhancement
# https://github.com/vbelz/Speech-enhancement/blob/master/model_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

## Module which does nothing 
class passing(nn.Module) : 
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return x

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
        else :
            raise Exception("ERROR::Encoder:Unknown activation type " + str(activation))

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.acti(x)
        return x

class Unet20(nn.Module):
    def __init__(self, hp,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros",
                 ):
        super().__init__()

        self.hp = hp
        input_channels = hp.model.UNET.channels
        dropout = hp.model.UNET.dropout
        activation = hp.model.UNET.activation
        mask_activation = hp.model.UNET.mask_activation

       
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
            self.mask_acti = passing()
        else :
            raise Exception('ERROR:Unknown activation : ' + str(activation))

        self.add_module("linear", linear)
        self.padding_mode = padding_mode

    def forward(self, x):        
        # ipnut : [ Batch Channel Freq Time]

        x_skip = []
        for i, encoder in enumerate(self.encoders):
            x_skip.append(x)
            x = encoder(x)
            #print("x{}".format(i), x.shape)
        # x_skip : x0=input x1 ... x9

        #print("fully encoded ",x.shape)
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
