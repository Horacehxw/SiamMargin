import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from os.path import realpath, dirname, join
from siam_margin.utils import get_subwindow_tracking, Average
import torch


def adjust_window(state, instance_size=255):
    p = state['p']
    if p.instance_size == instance_size:
        return
    p.instance_size = instance_size
    p.score_size = (instance_size-p.exemplar_size) // p.total_stride + 9
    p.anchor = p.anchors[0] if instance_size == p.default_instance_size else p.anchors[1]
    window = p.windows[0] if instance_size == p.default_instance_size else p.windows[1]
    state['p'] = p
    state['window'] = window

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    windowing = 'cosine'  
    exemplar_size = 127  
    instance_size = 255 
    default_instance_size = 255  
    total_stride = 8
    score_size = (instance_size-exemplar_size) // total_stride + 9
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    with torch.no_grad():
        delta, score = net(x_crop)

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1) # [2,5*17*17]
        score = F.softmax(score, dim=0).data[1, :].cpu().numpy()

        delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]


        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        best_pscore_id = np.argmax(pscore)

        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

        speed = delta[0, best_pscore_id]**2 + delta[1, best_pscore_id]**2
        speed = speed** 0.5

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        # update
        origin_target_pos = target[:2] * scale_z
        origin_target_sz = target_sz * scale_z
        net.update_roi_align(origin_target_pos,origin_target_sz,np.max(pscore))
    return target_pos, target_sz, score[best_pscore_id], speed


def SiamRPN_init(im, target_pos, target_sz, net, penalty=0.055, window=0.30, lr=0.295, bk_size=287):
    bk_size = int(bk_size)
    state = dict()
    p = TrackerConfig()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p.penalty_k = penalty
    p.window_influence = window
    p.lr = lr
    if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
        p.instance_size = bk_size  # small object big search region
    else:
        p.instance_size = p.default_instance_size

    p.score_sizes = [(p.default_instance_size - p.exemplar_size) // p.total_stride + 1 + 8, (bk_size - p.exemplar_size) // p.total_stride + 1 + 8]
    p.score_size = p.score_sizes[0] if p.instance_size == p.default_instance_size else p.score_sizes[1]

    p.anchors = [generate_anchor(p.total_stride, p.scales, p.ratios, p.score_sizes[0]), generate_anchor(p.total_stride, p.scales, p.ratios, p.score_sizes[1])]
    p.anchor = p.anchors[0] if p.instance_size == p.default_instance_size else p.anchors[1]

    state['speed'] = Average()

    avg_chans = np.mean(im, axis=(0, 1))
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    def window_helper(score_size):
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
        elif p.windowing == 'uniform':
            window = np.ones((score_size, score_size))
        window = np.tile(window.flatten(), p.anchor_num)
        return window

    p.windows = list(map(window_helper, p.score_sizes))
    window = p.windows[0] if p.instance_size == p.default_instance_size else p.windows[1]

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['last_move'] = np.array([0.,0.])
    return state


def SiamRPN_track(state, im, device=0):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    last_move = state['last_move']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) // 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    cur_target_pos, cur_target_sz, score, speed = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)

    if score < 0.1: # if low confidence
        move_target_pos = target_pos + 0.9 * last_move
        # One more shot
        x_crop = Variable(get_subwindow_tracking(im, move_target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
        cur_target_pos, cur_target_sz, score, speed = tracker_eval(net, x_crop.cuda(), move_target_pos, target_sz * scale_z, window, scale_z, p) # More

    cur_move = cur_target_pos - target_pos
    target_pos = cur_target_pos
    target_sz = cur_target_sz
    state['last_move'] = cur_move


    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score

    state['speed'].update(speed)
    # adjust search region
    if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004 or (state['speed'].max > 80):
        adjust_window(state, 351)
    else:
        adjust_window(state, 255)

    return state
