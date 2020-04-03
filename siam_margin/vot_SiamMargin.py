#!/usr/bin/python
from siam_margin import vot
from siam_margin.net import SiamRPN
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import sys
if not sys.platform=="linux2":
    sys.platform="linux2" # to import trax
import siam_margin.vot
from siam_margin.vot import Rectangle
sys.platform="linux"

import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from siam_margin.run_SiamRPN import SiamRPN_init, SiamRPN_track
from siam_margin.utils import get_axis_aligned_bbox, cxy_wh_2_rect
from collections import OrderedDict
from siam_margin.resnet import ResNet
import siam_margin.vot


print("=== Build Model ===")

lambda_u = 0.18
lambda_s = 0.60
window = 0.38
lr = 0.28
penalty = 0.04
bk_size = 351

net = SiamRPN(lambda_s=lambda_s, lambda_u=lambda_u)
net_file = join(realpath(dirname(__file__)), "model.pth")
weights = torch.load(net_file)
sd = net.state_dict()
for k in sd:
    if k in weights:
        sd[k] = weights[k]
    else:
        print("Warning : key '%s' is not in model weights!" % k)
net.load_state_dict(sd)
net.eval().cuda()

for parameter in net.parameters():
    parameter.volatile = True

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

imagefile = "".join(imagefile).split("'")[2] 

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(imagefile)  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net, window=window, lr=lr, penalty=penalty, bk_size=bk_size)  # init tracker
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    imagefile = "".join(imagefile).split("'")[2]
    im = cv2.imread(imagefile)  # HxWxC
    state = SiamRPN_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))

