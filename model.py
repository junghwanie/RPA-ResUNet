import torch.nn as nn
from utils import *


""" RPA-ResUNet definition """
class RPAResUNet(nn.Module):
    def __init__(self, num_classes):
        super(RPAResUNet, self).__init__()
        self.num_classes = num_classes

        """ Input block -> residual layer """
        self.residual_1_a = self.input_block(in_channels=3, out_channels=32)
        self.residual_1_b = self.input_skip(in_channels=3, out_channels=32)

        # input block -> spatial attention
        self.spatial_attention = SpatialAttention()

        """ Residual block x 3, Encoding """
        self.residual_2 = ResidualBlock(32, 64, 2, 1)
        self.residual_3 = ResidualBlock(64, 128, 2, 1)
        self.residual_4 = ResidualBlock(128, 256, 2, 1)

        """ Bridge block, ASPP """
        # 3 -> 32, 1024 -> 512, computation cost
        self.bridge_aspp = ASPP(256, 512)

        """ Last Encoder layer <- channel attention """
        self.channel_attention = ChannelAttention(512)

        """ Residual block x3, Decoding 1st block """
        # upsample out_feature 512 -> 256 ?
        self.upsample_block_1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0
        )
        self.residual_5 = ResidualBlock(512 + 128, 256, 1, 1)  # upsampling + residual_3

        """ 2nd Residual block, Decoder """
        self.upsample_block_2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0
        )
        self.residual_6 = ResidualBlock(256 + 64, 128, 1, 1)

        """ 3rd Residual block, Decoder """
        self.upsample_block_3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0
        )
        self.residual_7 = ResidualBlock(128 + 32, 64, 1, 1)

        """ output block """
        self.output_aspp = ASPP(64, 32)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
            nn.Sigmoid(),
        )

    def input_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        return block

    def input_skip(self, in_channels, out_channels):
        skip = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return skip

    def forward(self, X):
        residual_1_a_out = self.residual_1_a(X)
        residual_1_b_out = self.residual_1_b(X)
        residual_1_out = residual_1_a_out + residual_1_b_out
        spatial_attention_out = self.spatial_attention(residual_1_out) * residual_1_out

        residual_2_out = self.residual_2(spatial_attention_out)
        residual_3_out = self.residual_3(residual_2_out)
        residual_4_out = self.residual_4(residual_3_out)

        bridge_aspp_a = self.bridge_aspp(residual_4_out)
        channel_attention_out = self.channel_attention(bridge_aspp_a) * bridge_aspp_a

        """ decoder block without attention modules """

        upsample_1_out = self.upsample_block_1(channel_attention_out)
        residual_5_out = self.residual_5(
            torch.cat((upsample_1_out, residual_3_out), dim=1)
        )

        upsample_2_out = self.upsample_block_2(residual_5_out)
        residual_6_out = self.residual_6(
            torch.cat((upsample_2_out, residual_2_out), dim=1)
        )

        upsample_3_out = self.upsample_block_3(residual_6_out)
        residual_7_out = self.residual_7(
            torch.cat((upsample_3_out, residual_1_out), dim=1)
        )

        output_aspp_b = self.output_aspp(residual_7_out)
        out = self.output(output_aspp_b)
        return out