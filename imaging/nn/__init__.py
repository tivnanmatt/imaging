import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def pad_periodic(x, pad_size):
    x_pad = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+2*pad_size, x.shape[3]+2*pad_size).to(device)
    x_pad[:,:,pad_size:-pad_size,pad_size:-pad_size] = x
    # same as the other side
    x_pad[:,:,0:pad_size,pad_size:-pad_size] = x[:,:,-pad_size:,:]
    x_pad[:,:,-pad_size:,pad_size:-pad_size] = x[:,:,0:pad_size,:]
    x_pad[:,:,pad_size:-pad_size,0:pad_size] = x[:,:,:,-pad_size:]
    x_pad[:,:,pad_size:-pad_size,-pad_size:] = x[:,:,:,0:pad_size]
    # now the corners
    x_pad[:,:,0:pad_size,0:pad_size] = x[:,:,-pad_size:,-pad_size:]
    x_pad[:,:,-pad_size:,0:pad_size] = x[:,:,0:pad_size,-pad_size:]
    x_pad[:,:,0:pad_size,-pad_size:] = x[:,:,-pad_size:,0:pad_size]
    x_pad[:,:,-pad_size:,-pad_size:] = x[:,:,0:pad_size,0:pad_size]
    return x_pad



class ConvBlock(torch.nn.Module):
    def __init__(   self, 
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1,
                    periodic_pad=False
                    ):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.periodic_pad = periodic_pad
        self.kernel_size = kernel_size
    def forward(self, x):
        if self.periodic_pad:
            x = pad_periodic(x, self.kernel_size // 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DoubleConvBlock(torch.nn.Module):
    def __init__(   self, 
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1,
                    periodic_pad=False,
                    batch_norm=True,
                    activation_fn=torch.nn.ReLU()
                    ):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.act1 = activation_fn
        self.conv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.act2 = activation_fn
        self.periodic_pad = periodic_pad
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
    def forward(self, x):
        if self.periodic_pad:
            x = pad_periodic(x, self.kernel_size // 2)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act1(x)
        if self.periodic_pad:
            x = pad_periodic(x, self.kernel_size // 2)
        x = self.conv2(x)
        if self.periodic_pad:
            x = x[:,:,2*(self.kernel_size // 2):-2*(self.kernel_size // 2),2*(self.kernel_size // 2):-2*(self.kernel_size // 2)]
        if self.batch_norm:
            x = self.bn2(x)
        x = self.act2(x)
        return x
