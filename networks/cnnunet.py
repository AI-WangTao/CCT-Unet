import torch
import torch.nn as nn
from networks.swin_transformer import PatchEmbed
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, padding, dirate=1):  # dirate=1表示无空洞卷积
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=padding, dilation=1 * dirate)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, 3, padding=padding, dilation=1 * dirate)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class CNNUNET(nn.Module):
    def __init__(self,img_size=224,patch_size=4):
        super(CNNUNET, self).__init__()

        self.rebnconv_1d = REBNCONV(in_ch=3, out_ch=64, padding=1, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.rebnconv_2d = REBNCONV(in_ch=64, out_ch=128, padding=1, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.rebnconv_3d = REBNCONV(in_ch=128, out_ch=256, padding=1, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.rebnconv_4d = REBNCONV(in_ch=256, out_ch=512, padding=1, dirate=1)

        self.rebnconv_4up = REBNCONV(in_ch=512, out_ch=512, padding=1, dirate=1)

        self.conv_3up = nn.Conv2d(512, 256, 3, padding=1)
        self.rebnconv_3up = REBNCONV(in_ch=512, out_ch=256, padding=1, dirate=1)

        self.conv_2up = nn.Conv2d(256, 128, 3, padding=1)
        self.rebnconv_2up = REBNCONV(in_ch=256, out_ch=128, padding=1, dirate=1)

        self.conv_1up = nn.Conv2d(128, 64, 3, padding=1)
        self.rebnconv_1up = REBNCONV(in_ch=128, out_ch=64, padding=1, dirate=1)

        self.patchembed_1d = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=64, embed_dim=96)
        self.patchembed_2d = PatchEmbed(img_size=int(img_size/2), patch_size=patch_size, in_chans=128, embed_dim=192)
        self.patchembed_3d = PatchEmbed(img_size=int(img_size/4), patch_size=patch_size, in_chans=256, embed_dim=384)
        self.patchembed_4d = PatchEmbed(img_size=int(img_size/8), patch_size=patch_size, in_chans=512, embed_dim=768)

        self.patchembed_4up = PatchEmbed(img_size=int(img_size/8), patch_size=patch_size, in_chans=512, embed_dim=768)
        self.patchembed_3up = PatchEmbed(img_size=int(img_size/4), patch_size=patch_size, in_chans=256, embed_dim=384)
        self.patchembed_2up = PatchEmbed(img_size=int(img_size/2), patch_size=patch_size, in_chans=128, embed_dim=192)
        self.patchembed_1up = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=64, embed_dim=96)

    def upsample(self, src, tar):
        src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
        return src

    def cnnembed(self, x_d1, x_d2, x_d3, x_d4, x_up4, x_up3, x_up2, x_up1):
        x_d1_embed = self.patchembed_1d(x_d1)  # [2, 3136, 96]
        x_d2_embed = self.patchembed_2d(x_d2)  # [2, 784, 192]
        x_d3_embed = self.patchembed_3d(x_d3)
        x_d4_embed = self.patchembed_4d(x_d4)
        x_up4_embed = self.patchembed_4up(x_up4)
        x_up3_embed = self.patchembed_3up(x_up3)
        x_up2_embed = self.patchembed_2up(x_up2)  # [2, 784, 192]
        x_up1_embed = self.patchembed_1up(x_up1)  # [2, 3136, 96]
        return x_d1_embed, x_d2_embed, x_d3_embed, x_d4_embed, x_up4_embed, x_up3_embed, x_up2_embed, x_up1_embed

    def forward(self, x):
        '''CNN下采样'''
        x_d1 = self.rebnconv_1d(x)  # [2, 64, 224, 224]
        x_d2 = self.pool1(x_d1)  # [2, 64, 112,112]

        x_d2 = self.rebnconv_2d(x_d2)  # [2, 128, 112,112]
        x_d3 = self.pool2(x_d2)  # [2, 128, 56,56]

        x_d3 = self.rebnconv_3d(x_d3)  # [2, 256, 56,56]
        x_d4 = self.pool3(x_d3)  # [2, 256, 28,28]

        x_d4 = self.rebnconv_4d(x_d4)  # [2, 512, 28,28]

        '''CNN上采样'''
        x_up4 = self.rebnconv_4up(x_d4)
        x_up3 = self.rebnconv_3up(torch.cat((self.conv_3up(self.upsample(x_up4, x_d3)), x_d3), 1))  # [2, 256, 56, 56]
        x_up2 = self.rebnconv_2up(torch.cat((self.conv_2up(self.upsample(x_up3, x_d2)), x_d2), 1))  # [2, 128, 112, 112]
        x_up1 = self.rebnconv_1up(torch.cat((self.conv_1up(self.upsample(x_up2, x_d1)), x_d1), 1))  # [2, 64, 224, 224]
        x_embed = self.cnnembed(x_d1, x_d2, x_d3, x_d4, x_up4, x_up3, x_up2, x_up1)
        return x_embed



if __name__ == '__main__':
    input = torch.ones((2, 3, 224, 224))

    net = CNNUNET()
    x = net(input)
    for i in range(len(x)):
        print(x[i].shape)
