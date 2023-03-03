import torch 

from . import pad_periodic, ConvBlock, DoubleConvBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# make a unet
class Unet(torch.nn.Module):
    def __init__(   self, 
                    base_channels, 
                    in_channels, 
                    out_channels, 
                    kernel_size=3,
                    periodic_pad=False
                    ):
        super(Unet, self).__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.conv1 = DoubleConvBlock(in_channels, base_channels, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.down1 = ConvBlock(base_channels, base_channels*2, kernel_size=2, stride=2)
        self.conv2 = DoubleConvBlock(base_channels*2, base_channels*2, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.down2 = ConvBlock(base_channels*2, base_channels*4, kernel_size=2, stride=2)
        self.conv3 = DoubleConvBlock(base_channels*4, base_channels*4, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.down3 = ConvBlock(base_channels*4, base_channels*8, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(base_channels*8, base_channels*8, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.down4 = ConvBlock(base_channels*8, base_channels*16, kernel_size=2, stride=2)
        self.conv5 = DoubleConvBlock(base_channels*16, base_channels*16, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.up5 = torch.nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.conv6 = DoubleConvBlock(base_channels*16, base_channels*8, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.up6 = torch.nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.conv7 = DoubleConvBlock(base_channels*8, base_channels*4, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.up7 = torch.nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.conv8 = DoubleConvBlock(base_channels*4, base_channels*2, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.up8 = torch.nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.conv9 = DoubleConvBlock(base_channels*2, base_channels, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad)
        self.conv10 = torch.nn.Conv2d(base_channels, out_channels, kernel_size, padding='same')
    def forward(self, x):
        x1 = self.conv1(x)
        x1_down = self.down1(x1)
        x2 = self.conv2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.conv3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.conv4(x3_down)
        x4_down = self.down4(x4)
        x5 = self.conv5(x4_down)
        x5_up = self.up5(x5)
        x6 = torch.cat([x4, x5_up], dim=1)
        x6 = self.conv6(x6)
        x6_up = self.up6(x6)
        x7 = torch.cat([x3, x6_up], dim=1)
        x7 = self.conv7(x7)
        x7_up = self.up7(x7)
        x8 = torch.cat([x2, x7_up], dim=1)
        x8 = self.conv8(x8)
        x8_up = self.up8(x8)
        x9 = torch.cat([x1, x8_up], dim=1)
        x9 = self.conv9(x9)
        x10 = self.conv10(x9)
        return x10


# make a unet
class ConvolutionalPredictor(torch.nn.Module):
    def __init__(   self, 
                    base_channels, 
                    in_channels, 
                    out_channels, 
                    kernel_size=3,
                    periodic_pad=False
                    ):
        super(ConvolutionalPredictor, self).__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.conv1 = DoubleConvBlock(in_channels, base_channels, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad, activation_fn=torch.nn.LeakyReLU(negative_slope=0.1))
        self.down1 = ConvBlock(base_channels, base_channels*2, kernel_size=2, stride=2)
        self.conv2 = DoubleConvBlock(base_channels*2, base_channels*2, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad, activation_fn=torch.nn.LeakyReLU(negative_slope=0.1))
        self.down2 = ConvBlock(base_channels*2, base_channels*4, kernel_size=2, stride=2)
        self.conv3 = DoubleConvBlock(base_channels*4, base_channels*4, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad, activation_fn=torch.nn.LeakyReLU(negative_slope=0.1))
        self.down3 = ConvBlock(base_channels*4, base_channels*8, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(base_channels*8, base_channels*8, kernel_size=kernel_size, stride=1, periodic_pad=periodic_pad, activation_fn=torch.nn.LeakyReLU(negative_slope=0.1))
        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(base_channels*8*8*8, base_channels*8*8)
        self.fc2 = torch.nn.Linear(base_channels*8*8, base_channels*8)
        self.fc3 = torch.nn.Linear(base_channels*8, base_channels)
        self.fc4 = torch.nn.Linear(base_channels, out_channels)
        self.activation_fn = torch.nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_down = self.down1(x1)
        x2 = self.conv2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.conv3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.conv4(x3_down)
        x4_flat = self.flat(x4)
        x5 = self.fc1(x4_flat)
        x6 = self.fc2(x5)
        x7 = self.fc3(x6)
        x8 = self.fc4(x7)
        # softmax activation
        # x8 = torch.nn.functional.softmax(x8)
        return x8

class ConvolutionalMLP(torch.nn.Module):
    def __init__(   self, 
                    base_channels, 
                    in_channels, 
                    out_channels
                    ):
        super(ConvolutionalMLP, self).__init__()
        self.conv1 = ConvBlock(in_channels, base_channels*16, 1)
        self.conv2 = ConvBlock(base_channels*16, base_channels*8, 1)
        self.conv3 = ConvBlock(base_channels*8, base_channels*4, 1)
        self.conv4 = ConvBlock(base_channels*4, base_channels*2, 1)
        self.conv5 = ConvBlock(base_channels*2, base_channels, 1)
        self.conv6 = torch.nn.Conv2d(base_channels, out_channels, 1, padding='same')
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x



class TuneableUnet(torch.nn.Module):
    def __init__(self,
                aux_decoder_base_channels=64,
                aux_decoder_in_channels=1,
                unet_base_channels=64,
                unet_in_channels=32,
                image_size=64,
                periodic_pad=False,
                 ):
        super(TuneableUnet, self).__init__()
        self.aux_decoder = ConvolutionalMLP(   base_channels=aux_decoder_base_channels, 
                                                in_channels=aux_decoder_in_channels, 
                                                out_channels=unet_in_channels-1).to(device)
        self.unet = Unet(   base_channels=unet_base_channels, 
                            in_channels=unet_in_channels, 
                            out_channels=1,
                            periodic_pad=periodic_pad).to(device)
        self.image_size = image_size

    def forward(    self, 
                    y, 
                    aux_inputs
                    ):
        aux_dec = self.aux_decoder(aux_inputs).repeat(1, 1, self.image_size, self.image_size)
        x_t_and_y_and_t_dec = torch.cat([y, aux_dec], dim=1)
        x_hat = self.unet(x_t_and_y_and_t_dec)
        return x_hat
