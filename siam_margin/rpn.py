import torch.nn as nn
import torch.nn.functional as F

def conv2d_dw_group(x, kernel):
    batch, channel = x.shape[:2]
    k_channel = kernel.shape[1]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(
        batch * k_channel, 1, kernel.size(2),
        kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, -1, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(
                inplace=True),
             )
        self.conv_search = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(
                inplace=True),
             )

        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


class L_DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(L_DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden*2,
                kernel_size=kernel_size,
                bias=False),
            )
        self.conv_search = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                bias=False),
            )

        self.head = nn.Sequential(
                nn.Conv2d(hidden*2, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )


    def forward_corr(self, kernel, input):
        if self.kernel is None:
            assert False, "error!"
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, self.kernel)
        return feature
    def make_kernel(self, z_f):
        return self.conv_kernel(z_f)

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


class RPN(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(RPN, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = L_DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return loc, cls
