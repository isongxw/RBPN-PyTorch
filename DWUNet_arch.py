import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
# from models.modules.PixelShuffleModel import PSUpsample


def unified_scale(x1, anchor):
    diffY = anchor.size()[2] - x1.size()[2]
    diffX = anchor.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    return x1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Resiudual Dense Block
class RDB(nn.Module):
    def __init__(self, channel_in):
        super(RDB, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=channel_in, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(
            in_channels=channel_in * 3, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)

        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)

        out = self.relu3(self.conv_3(conc))

        out = torch.add(out, residual)

        return out


class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()

        self.wavelet = DWTForward(J=1, wave='haar')

        # self.dcr = nn.Sequential(RDB(channel_in),
        #                          RDB(channel_in))
        self.mid = nn.Conv2d(
            in_channels=channel_in, out_channels=channel_in, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Conv2d(channel_in * 4, channel_out,
                              kernel_size=3, stride=1, padding=1)

    def transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def forward(self, x):
        yl, yh = self.wavelet(x)  # downsample channel_in
        # yl = self.dcr(yl)  # channel_in
        yl = self.mid(yl)  # channel_
        out = self.transformer(yl, yh)  # channel_in * 4

        return self.conv(out)


class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()

        self.wavelet_i = DWTInverse(wave='haar')
        # self.dcr = nn.Sequential(RDB(channel_in),
        #                          RDB(channel_in))
        self.conv1 = nn.Conv2d(channel_in // 4, channel_in //
                               2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_in, channel_out,
                               kernel_size=3, stride=1, padding=1)

    def transformer_i(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh

    def forward(self, x1, x2):
        out = self.transformer_i(x2)  # channel_in // 4
        out = self.wavelet_i(out)
        out = self.conv1(out)
        out = torch.cat([unified_scale(out, x1), x1], 1)
        # out = self.dcr(out)

        return self.conv2(out)


class DWUNet(nn.Module):
    def __init__(self):
        super(DWUNet, self).__init__()

        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.inc = DoubleConv(256, 64)
        self.down1 = Down(64, 128)  # 128*128
        self.down2 = Down(128, 256)  # 64*64
        self.down3 = Down(256, 512)  # 32*32

        self.up1 = Up(512, 256)  # 32*32
        self.up2 = Up(256, 128)  # 64*64
        self.up3 = Up(128, 64)  # 128*128

        self.outc = nn.Conv2d(in_channels=64, out_channels=3,
                              kernel_size=3, stride=1, padding=1)
        self.relu = nn.PReLU()
        # self.convf = nn.Conv2d(
        #     in_channels=3, out_channels=3 * 16, kernel_size=3, stride=1, padding=1)
        # self.upsample = PSUpsample()

    def forward(self, x):
        x = self.inc(x)
        x = self.upsample(x)
        x1 = self.upsample(x)
        residual = x1

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x3, x4)
        x = self.up2(x2, x)
        out = self.up3(x1, x)
        # out = self.outc(x)
        out = torch.add(self.relu(unified_scale(out, residual)), residual)

        # out = torch.sigmoid(self.convf(out))

        return out
