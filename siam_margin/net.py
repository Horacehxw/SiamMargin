import torch
import torch.nn as nn
from siam_margin.rpn import RPN
from siam_margin.resnet import ResNet
import numpy as np
from torch.autograd import Variable


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l, r = 4, -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(nn.Module):
    def __init__(self):
        super(ResDown, self).__init__()
        self.features = ResNet()
        self.downsample = ResDownS(1024, 256) # 1024 for layer 3, 2048 for layer 4, 512 for layer 2

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output)
        return p3


class SiamRPN(nn.Module):
    def __init__(self, config=None, feature_in=256, feature_out=256, anchor=5, lambda_u=0.0, lambda_s=1.0):
        super(SiamRPN, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.feature = ResDown()
        self.rpn = RPN(anchor, feature_in, feature_out)
        self.config = config
        if config is not None :
            self.batch_size = config.batch_size / config.gpu_num
        else :
            self.batch_size = 1
        self.z_f = None
        self.first_kernel = None
        self.MA_kernel = None

        self.lambda_s = lambda_s
        self.lambda_u = lambda_u

    def _fix_layers(self):
        def _fix(layer):
            for param in layer.parameters():
                param.requires_grad = False

        _fix(self.feature.features)

    def get_backbone(self):
        return self.feature.features

    def get_head(self):
        return self.rpn

    def _unfix_layers(self):
        for param in self.feature.features.parameters():
            param.requires_grad = True

    def forward(self,x,z=None):
        if z is None:
            z_f = self.z_f
        else:
            z_f = self.feature(z)
        self.x_f = self.feature(x)
        return self.rpn(z_f, self.x_f)

    def search(self, x):
        x_f = self.feature(x)
        return self.rpn(self.z_f, x_f)

    def temple(self, z):
        self.z_f = self.feature(z)
        self.rpn.cls.kernel = self.rpn.cls.make_kernel(self.z_f)
        self.first_kernel = self.rpn.cls.kernel
        self.MA_kernel = self.rpn.cls.kernel

    def update_roi_align(self,target_pos,target_sz,score):
        lambda_u = self.lambda_u * float(score)
        lambda_s = self.lambda_s
        N, C, H, W = self.x_f.shape
        stride = 8
        assert N == 1, "not supported"
        l = W // 2
        x = range(-l,l+1)
        y = range(-l,l+1)

        hc_z = (target_sz[1] + 0.3 * sum(target_sz)) / stride
        wc_z = (target_sz[0] + 0.3 * sum(target_sz)) / stride
        grid_x = np.linspace(- wc_z/2, wc_z/2, 17)
        grid_y = np.linspace(- hc_z/2, hc_z/2, 17)
        grid_x = grid_x[5:-5] + target_pos[0] / stride
        grid_y = grid_y[5:-5] + target_pos[1] / stride
        x_offset = grid_x/l
        y_offset = grid_y/l

        grid = np.reshape(np.transpose([np.tile(x_offset,len(y_offset)), np.repeat(y_offset,len(x_offset))]), (len(grid_y), len(grid_x), 2))
        grid = Variable(torch.from_numpy(grid).unsqueeze(0)).cuda()

        zmap = nn.functional.grid_sample(self.x_f.data.double(), grid).float()
        cls_kernel = self.rpn.cls.make_kernel(zmap)
        self.MA_kernel = (1-lambda_u) * self.MA_kernel + lambda_u * cls_kernel
        self.rpn.cls.kernel = self.first_kernel * lambda_s + self.MA_kernel*(1.0-lambda_s)
