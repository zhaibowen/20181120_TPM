# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 9:16
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : utils.py
import time
import numpy as np
from contextlib import contextmanager
from config import Exact_name


def dt_str():
    return time.strftime('%Y.%m.%d', time.localtime(time.time()))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def parallel_predict(sample, func, TPM):
    """parallel predict samples or generate features"""
    if isinstance(TPM, str) and TPM == Exact_name:
        TPM = sample['TPM']
    preds = func(measure=sample['measure'], TPM=TPM)
    sample.update(preds)
    return sample


def frame_expand(data):
    """data [sample, frame, ?] --> [sample * frame, frame * ?]"""
    sample = data.shape[0]
    frame = data.shape[1]
    result = np.zeros([frame, *data.shape], dtype=np.float32)
    for i in range(frame):
        result[i, :, -i-1:] = data[:, :i+1]
    return result[1:].transpose([1, 0]+list(range(len(result.shape)))[2:]).reshape([(frame-1)*sample, -1])
