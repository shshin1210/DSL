import array
import os
import re
import torch
import numpy as np

cmf_path = './data/judd_vos.txt'

cmf = np.loadtxt(cmf_path, usecols=(1, 2, 3))

illum_path = './data/d65.txt'
illum = np.loadtxt(cmf_path, usecols=(1, ))

d65_xyz = np.array([0.95047, 1.0, 1.08883])

M = np.array([[3.2406, -1.5372, -0.4986],
              [-0.9689, 1.8758, 0.0415],
              [0.0557, -0.2040, 1.0570]])


def xyz2srgb(xyz, gamma=2.2):
    sRGB = np.tensordot(xyz, M.transpose(), [2, 0])
    sRGB[sRGB > 1] = 1
    sRGB[sRGB < 0] = 0

    RGB = sRGB ** (1 / gamma)

    RGB[RGB > 1] = 1
    RGB[RGB < 0] = 0

    return RGB


def spec_to_xyz(spectrum: np.ndarray,
                wvls: np.ndarray,
                illuminant: np.ndarray):
    cmf_indicies = (np.squeeze(wvls).astype(np.uint32) - 380) // 1
    illum_indices = (np.squeeze(wvls).astype(np.uint32) - 380) // 1
    cmf_matrix = cmf[cmf_indicies, :]
    if cmf_matrix.ndim == 1:
        cmf_matrix = np.expand_dims(cmf_matrix, 0)
    illum_matrix = illuminant[illum_indices]
    matrix = cmf_matrix.T# * illum_matrix.T

    # if spectrum.shape[-1] != matrix.T.shape[-1]
    #     matrix

    xyz = (spectrum @ matrix.T)

    # xyz = xyz / np.sum(xyz)
    k = 100 / (cmf_matrix[:, 1] * illum_matrix).sum()
    xyz = xyz * k
    xyz[xyz < 0] = 0
    return xyz


def whitebalance(ref_xyz: np.ndarray):
    mmean = ref_xyz.mean(axis=(0, 1))
    maxY = ref_xyz[:, :, 1].max()
    ref_xyzw = mmean * (maxY / mmean[1])
    # ref_xyzw = mmean * (1 / mmean[1])
    return ref_xyz / ref_xyzw


# rad_x: numpy (N,M,C), tensor (C,N,M)
# wvls: [400, 410, ..., 700] nm
def hs2srgb(rad_x, wvls,
            refwhite=None, illuminant=illum, output_mode_HWC=False,
            gamma=2.2):


    if type(rad_x) == torch.Tensor:
        rad_x = rad_x.data.cpu().numpy()
        rad_x = np.transpose(rad_x, (1, 2, 0))


    rad_x[rad_x < 0] = 0
    rad_x[rad_x > 1] = 1
    if refwhite is not None:
        refco = 0.9781
        refwhite2 = refwhite / refco

        ref_x = rad_x / refwhite2
        whiteref = refwhite / refwhite2
    else:
        ref_x = rad_x

    ref_xyz = spec_to_xyz(ref_x, wvls, illuminant)

    do_WB = True
    if do_WB:
        if refwhite is not None:
            white_xyz = spec_to_xyz(whiteref, wvls, illuminant)
            ref_xyzw = ref_xyz / white_xyz
        else:
            ref_xyzw = whitebalance(ref_xyz)
    else:
        ref_xyzw = ref_xyz

    wpD65 = d65_xyz
    ref_xyzw = ref_xyzw * wpD65

    srgb = xyz2srgb(ref_xyzw, gamma)
    if output_mode_HWC:
        return srgb
    else:
        return np.transpose(srgb, (2,0,1))


def hs2rgb(rad_x, wvls, crf, output_mode_HWC=False, gamma=2.2, normalize=False, WB=True, rawRGB2sRGB_mat=None, vmin=0,vmax=1):
    # conversion_mat: K x 3, where K is the number of the hyperspectral channels
    # rad_x: R x C x N, where N is the numer of spectral channels
    # crf: N x 3
    # wvls: N x 1

    if type(rad_x) == torch.Tensor:
        # nCh, R, C
        rad_x = rad_x.data.cpu().numpy()
        rad_x = np.transpose(rad_x, (1, 2, 0))

    rgb = np.zeros((rad_x.shape[0], rad_x.shape[1], 3))
    for i in range(len(wvls)):
        rgb[:,:,0] += rad_x[:,:,i] * crf[i, 0]
        rgb[:,:,1] += rad_x[:,:,i] * crf[i, 1]
        rgb[:,:,2] += rad_x[:,:,i] * crf[i, 2]


    if WB:
        rgb_sum = crf.sum(0)
        rgb[:, :, 0] /= rgb_sum[0]
        rgb[:, :, 1] /= rgb_sum[1]
        rgb[:, :, 2] /= rgb_sum[2]

    # ==============
    # raw RGB 2 sRGB
    if rawRGB2sRGB_mat is not None:
        [R,C,_]=rgb.shape
        rgb = (rawRGB2sRGB_mat@(rgb.reshape([R*C,3]).transpose((1,0)))).transpose((1,0)).reshape([R,C,3])
    # ==============

    if normalize:
        rgb = (rgb-vmin)/(vmax-vmin)

    rgb[rgb > 1] = 1
    rgb[rgb < 0] = 0

    rgb = rgb**(1/gamma)

    if output_mode_HWC:
        return rgb
    else:
        return np.transpose(rgb, (2,0,1))

