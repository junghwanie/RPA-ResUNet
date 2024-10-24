import torch.nn as nn
import torch
import torch.nn.functional as F


""" UNet definition. """
class UNet(nn.Module):
    def cont_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.Dropout2d(), # p=0.5
                              nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(out_channels))
        return block
    
    def expn_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channel,
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.Dropout2d(),
                              nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel,
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(mid_channel),
                              nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=2, padding=1, output_padding=1))
        return block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=mid_channel,
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.Dropout2d(),
                              nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel,
                                        kernel_size=kernel_size, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(mid_channel),
                              nn.Conv2d(in_channels=mid_channel, out_channels=out_channels,
                                        kernel_size=kernel_size, padding=1),
                              nn.Sigmoid()
                              )
        return block
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.contl_1 = self.cont_block(in_channels=in_channel, out_channels=64)
        self.contl_1_mp = nn.MaxPool2d(kernel_size=2)
        self.contl_2 = self.cont_block(in_channels=64, out_channels=128)
        self.contl_2_mp = nn.MaxPool2d(kernel_size=2)
        self.contl_3 = self.cont_block(in_channels=128, out_channels=256)
        self.contl_3_mp = nn.MaxPool2d(kernel_size=2)
        self.contl_4 = self.cont_block(in_channels=256, out_channels=512)
        self.contl_4_mp = nn.MaxPool2d(kernel_size=2)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, 
                                                  kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(1024),
                                        nn.Conv2d(in_channels=1024, out_channels=1024, 
                                                  kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(1024),
                                        nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                                           kernel_size=3, stride=2, padding=1, output_padding=1)
                                        )
        
        self.expnl_1 = self.expn_block(1024, 512, 256)
        self.expnl_2 = self.expn_block(512, 256, 128)
        self.expnl_3 = self.expn_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
    
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        contl_1_out = self.contl_1(x)
        contl_1_pool = self.contl_1_mp(contl_1_out)
        contl_2_out = self.contl_2(contl_1_pool)
        contl_2_pool = self.contl_2_mp(contl_2_out)
        contl_3_out = self.contl_3(contl_2_pool)
        contl_3_pool = self.contl_3_mp(contl_3_out)
        contl_4_out = self.contl_4(contl_3_pool)
        contl_4_pool = self.contl_4_mp(contl_4_out)
        
        bottleneck_out = self.bottleneck(contl_4_pool)
        
        """ Skip connection of expansive block. """
        expnl_1_cat = self.crop_and_concat(bottleneck_out, contl_4_out)
        expnl_1_out = self.expnl_1(expnl_1_cat)
        expnl_2_cat = self.crop_and_concat(expnl_1_out, contl_3_out)
        expnl_2_out = self.expnl_2(expnl_2_cat)
        expnl_3_cat = self.crop_and_concat(expnl_2_out, contl_2_out)
        expnl_3_out = self.expnl_3(expnl_3_cat)
        final_cat = self.crop_and_concat(expnl_3_out, contl_1_out)
        final_out = self.final_layer(final_cat)
        
        return final_out
    

""" UNet official code : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    Full assembly of the parts to form the complete network. """

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
    
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class build_unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(build_unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)