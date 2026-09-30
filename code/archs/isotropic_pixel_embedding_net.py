import torch
import torch.nn as nn


class MyPixelEmbeddingNet(nn.Module):
    def __init__(self, dim=16):
        super().__init__()

        # down
        self.d1_conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.d1_reluconv1 = nn.ReLU()

        self.pool1 = nn.AvgPool2d(2)
        self.d2_conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.d2_reluconv1 = nn.ReLU()

        self.pool2 = nn.AvgPool2d(2)
        self.d3_conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.d3_reluconv1 = nn.ReLU()

        # up
        self.u1_tconv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.u1_conv1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.u2_tconv = nn.ConvTranspose2d(8, 4, kernel_size=2,stride=2)
        self.u2_conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        # out
        self.conv_out = nn.Conv2d(8, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x: bs x channels x width x height 
        d1 = self.d1_reluconv1(self.d1_conv1(x))
        d2 = self.d2_reluconv1(self.d2_conv1(self.pool1(d1)))
        d3 = self.d3_reluconv1(self.d3_conv1(self.pool2(d2)))
        #print("d2: ", d2.size())
        #print("d3: ", d3.size())
        upconv1= self.u1_tconv(d3)
        #print("upconv1: ", upconv1.size())
        u1 = self.u1_conv1(torch.cat((upconv1, d2),dim=1))
        u2 = self.u2_conv1(torch.cat((self.u2_tconv(u1), d1),dim=1))

        out = self.conv_out(u2)
        #bs, channels, height, width = out.size()
        #print("out: ", out.size())
        #out = out.permute(0,2,3,1)
        #print("out_permuted: ", out.size())
        #out = out.reshape(bs, height * width, channels)
        #print("out_reshaped: ", out.size())
        return out
