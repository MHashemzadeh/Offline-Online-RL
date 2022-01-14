import numpy as np
import torch
# import cv2
# import copy
# from scipy.ndimage.filters import gaussian_filter


def random_shift(imgs, pad=1, prob=1.): ## for image based states
    t = b = c = 1
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        h, w = imgs.shape
    elif shape_len == 3:
        c, h, w = imgs.shape
    elif shape_len == 4:
       b, c, h, w = imgs.shape
#         b, h, w, c = imgs.shape
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same crop to all T
        imgs = imgs.transpose(1, 0, 2, 3, 4)
        _c = c
        c = t * c
        # imgs = imgs.reshape(b, t * c, h, w)
    imgs = imgs.reshape(b, c, h, w)

    crop_h = h
    crop_w = w

    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    b, c, h, w = padded.shape

    h_max = h - crop_h + 1
    w_max = w - crop_w + 1
    h1s = np.random.randint(0, h_max, b)
    w1s = np.random.randint(0, w_max, b)
    if prob < 1.:
        which_no_crop = np.random.rand(b) > prob
        h1s[which_no_crop] = pad
        w1s[which_no_crop] = pad

    shifted = np.zeros_like(imgs)
    for i, (pad_img, h1, w1) in enumerate(zip(padded, h1s, w1s)):
        shifted[i] = pad_img[:, h1:h1 + crop_h, w1:w1 + crop_w]

    if shape_len == 2:
        shifted = shifted.reshape(crop_h, crop_w)
    elif shape_len == 3:
        shifted = shifted.reshape(c, crop_h, crop_w)
    elif shape_len == 5:
        shifted = shifted.reshape(b, t, _c, crop_h, crop_w)
        shifted = shifted.transpose(1, 0, 2, 3, 4)

    return torch.tensor(shifted)

def random_amplitude_scaling(states, alpha=0., beta=1., prob=1.0, multivariate=False):  ## for vector based states
    if np.random.rand() < prob:
        if multivariate:
            states = states * np.random.uniform(alpha, beta, size=len(states))
        else:
            states = states * np.random.uniform(alpha, beta)
    return torch.tensor(states)
    