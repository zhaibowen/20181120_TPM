# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 9:04
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : track_generator.py
import numpy as np
from multiprocessing import Pool
from functools import partial
from config import processor


def one_track_generator(frame, acc, trans, noits, args):
    """
    generate one track
    according to 'online bayesian estimation of transition probabilities for markovian jump system' Jilkov and Li.
    [p, v]'(k) = [1, T; 0, 1][p, v]'(k-1) + [T^2/2, T]'[a(k)+w(k)]
    z(k) = p(k) + v(k)
    :param frame: the frame nums of the track 60
    :param acc: accelerate mode [0,-20,20]
    :param trans: transition matrix
    :param noits: noise transition matrix
    :param args:
        TPM: transition matrix shape [3,3]
        p0: initial position
        v0: initial velocity
        Wk: process noise
        Vk: measurement noise
        trans_rand: random number that decide mode of each moment
        u0: initial mode (0,1,2)
    :return:
    """
    TPM, p0, v0, Wk, Vk, trans_rand, u0 = args
    track = np.empty([frame, 2])
    track[0, :] = [p0, v0]
    TPM_cum = np.cumsum(TPM, axis=1)
    mode = np.empty(frame)
    uk = u0
    mode[0] = u0
    for k in range(1, frame):
        uk = np.minimum(np.searchsorted(TPM_cum[uk], trans_rand[k]), 2)
        mode[k] = uk
        track[k, :] = np.dot(trans, track[k-1, :]) + np.dot(noits, acc[uk]+Wk[k])
    measure = track[:, 0] + Vk
    return {'TPM': TPM, 'mode': mode, 'track': track, 'measure': measure}


def track_generator(sample_num, frame, acc, trans, noits, u0, p0_prob, v0_prob, wk_prob, vk_prob):
    TPM = np.random.rand(sample_num, 3, 3)  # random generate 1000 TPM
    TPM = TPM/TPM.sum(axis=2, keepdims=True)  # normalize
    p0 = p0_prob[0] + np.random.randn(sample_num)*p0_prob[1]  # p(0) ~ N(80000, 100^2)
    v0 = v0_prob[0] + np.random.randn(sample_num)*v0_prob[1]  # v(0) ~ N(80000, 100^2)
    Wk = wk_prob[0] + np.random.randn(sample_num, frame)*wk_prob[1]  # w(k) ~ N(0, 2^2)
    Vk = vk_prob[0] + np.random.randn(sample_num, frame)*vk_prob[1]  # v(k) ~ N(0, 100^2)
    trans_rand = np.random.rand(sample_num, frame)  # random numbers decide the mode of each frame
    u0 = np.searchsorted(np.cumsum(u0), np.random.rand(sample_num))
    with Pool(processes=processor) as pool:
        samples = pool.map(partial(one_track_generator, frame, acc, trans, noits),
                           zip(list(TPM), list(p0), list(v0), list(Wk), list(Vk), list(trans_rand), list(u0)))
    return samples
