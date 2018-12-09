# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 9:11
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : Metrics.py
import numpy as np
import matplotlib.pyplot as plt
from config import Exact_name, NonAdaptive_name, NIAdaptive_name, MyAdaptive_name, total_frame, AIAdaptive_name

UpperLeft = [100, 30, 580, 494]
UpperRight = [800, 30, 580, 494]
LowerLeft = [100, 550, 580, 494]
LowerRight = [800, 550, 580, 494]

name_list = [Exact_name, NIAdaptive_name, MyAdaptive_name, AIAdaptive_name, NonAdaptive_name]
line_list = ['', ':', '--', '-.', '-']
color_list = ['', 'g', 'r', 'purple', 'orange']


def position_MAE(data):
    plt.figure(1)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(*UpperLeft)
    for name, line, color in zip(name_list, line_list, color_list):
        if f'{name} est_state' in data[0].keys():
            AE = np.empty([len(data), total_frame])
            for i in range(len(data)):
                AE[i] = np.abs(data[i]['track'][:, 0] - data[i][f'{name} est_state'][:, 0])
            if name == Exact_name:
                plt.plot(range(1, 61), AE.mean(axis=0)[1:], marker='o', markerfacecolor='None',
                         markeredgewidth=1.5, markersize=3.5, label=Exact_name)
            else:
                plt.plot(range(1, 61), AE.mean(axis=0)[1:], color=color, linestyle=line, label=name)
    plt.title('Position MAE')
    plt.legend()


def velocity_MAE(data):
    plt.figure(2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(*UpperRight)
    for name, line, color in zip(name_list, line_list, color_list):
        if f'{name} est_state' in data[0].keys():
            AE = np.empty([len(data), total_frame])
            for i in range(len(data)):
                AE[i] = np.abs(data[i]['track'][:, 1] - data[i][f'{name} est_state'][:, 1])
            if name == Exact_name:
                plt.plot(range(1, 61), AE.mean(axis=0)[1:], marker='o', markerfacecolor='None',
                         markeredgewidth=1.5, markersize=3.5, label=Exact_name)
            else:
                plt.plot(range(1, 61), AE.mean(axis=0)[1:], color=color, linestyle=line, label=name)
    plt.title('velocity MAE')
    plt.legend()


def Mean_true_mode_probability(data):
    plt.figure(3)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(*LowerLeft)
    for name, line, color in zip(name_list, line_list, color_list):
        if f'{name} est_state' in data[0].keys():
            Counter = np.empty([len(data), total_frame], dtype=np.int32)
            for i in range(len(data)):
                Counter[i] = (data[i]['mode'] == data[i][f'{name} est_mode']).astype(np.int32)
            if name == Exact_name:
                plt.plot(range(1, 61), Counter.mean(axis=0)[1:], marker='o', markerfacecolor='None',
                         markeredgewidth=1.5, markersize=3.5, label=Exact_name)
            else:
                plt.plot(range(1, 61), Counter.mean(axis=0)[1:], color=color, linestyle=line, label=name)
    plt.title('Mean True-Mode Probability')
    plt.legend()


def TPM_MAE(data):
    plt.figure(4)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(*LowerRight)
    AE = {}
    for name in name_list:
        if f'{name} TPM' in data[0].keys():
            AE[name] = np.zeros([total_frame, 3, 3])
            for i in range(len(data)):
                AE[name] += np.abs(data[i][f'{name} TPM'] - data[i]['TPM'])
            AE[name] /= len(data)  # shape [61,3,3]
    for row in range(3):
        for col in range(3):
            plt.subplot(3, 3, row*3+col+1)
            for name, line, color in zip(name_list, line_list, color_list):
                if f'{name} TPM' in data[0].keys():
                    plt.plot(range(1, 61), AE[name][1:, row, col], color=color,
                             linestyle=line, label='%s $\pi_{%d%d}$' % (name, row+1, col+1))
            plt.legend()
    plt.suptitle(f'MAE of TPM Estimate', y=0.92)
    plt.legend()
