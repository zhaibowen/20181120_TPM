# -*- coding: utf-8 -*-
# @Time    : 2018/11/25 19:28
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : Filters.py
import copy
import numpy as np
from nn_model import NNModel


class BasicFilter:
    def __init__(self, name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame):
        self.name = name
        self.acc = acc
        self.G = trans
        self.B = noits
        self.H = meats
        self.u0 = u0
        self.v0_prob = v0_prob
        self.Wk = wk_prob[1]**2  # covariance of process noise, shape [1]
        self.Vk = vk_prob[1]**2  # covariance of measurement noise, shape [1]
        self.frame = total_frame
        # initial covariance of each mode, shape [3,2,2]
        self.P0 = np.tile(np.array([[vk_prob[1]**2, 0], [0, v0_prob[1]**2]], dtype=np.float32), [3, 1, 1])

    def initial_state(self, measure):
        mode = np.empty(self.frame, dtype=np.int32)
        mode[0] = np.argmax(self.u0)
        state = np.empty([self.frame, 2])
        state[0] = np.array([measure, self.v0_prob[0]])
        X = np.tile(state[0], [3, 1])  # estimation of each mode, shape [3,2]
        P = self.P0.copy()  # covariance of each mode, shape [3,2,2]
        LH = np.empty(3)  # mode likelihood, shape [3]
        uk = copy.copy(self.u0)  # mode probabllity of each frame, shape [3]
        return mode, state, X, P, LH, uk

    def IMMFilter(self, measure, TPM, X, P, LH, uk):
        # state mixing
        u_q_r = (TPM.T*uk).T
        u_q_r = u_q_r/u_q_r.sum(axis=0)  # transition probability from q to r, shape [3,3]
        Xo = np.dot(u_q_r.T, X)  # estimation of each mode after mixing, shape [3,2]
        Po = np.zeros_like(P)  # estimation of each mode after mixing, shape [3,2,2]
        for i in range(3):
            for j in range(3):
                Po[j] += u_q_r[i, j]*(P[i]+np.dot((X[i]-Xo[j]).reshape([2, 1]), (X[i]-Xo[j]).reshape([1, 2])))
        # kalman filter
        for i in range(3):
            P[i] = np.linalg.multi_dot([self.G, Po[i], self.G.T]) + \
                   self.Wk*np.dot(self.B.reshape([2, 1]), self.B.reshape([1, 2]))
            S = np.linalg.multi_dot([self.H, P[i], self.H])+self.Vk  # residuals covariance, shape [1]
            K = np.dot(P[i], self.H)/S  # shape [2]
            P[i] = np.dot(np.eye(2)-np.dot(K.reshape([2, 1]), self.H.reshape([1, 2])), P[i])
            V = measure-np.dot(self.H, np.dot(self.G, Xo[i])+self.B*self.acc[i])  # residuals, shape [1]
            X[i] = np.dot(self.G, Xo[i]) + self.B*self.acc[i] + K*V
            LH[i] = np.exp(-0.5*V**2/S)/np.sqrt(2*np.pi*S)
        LH /= np.sum(LH)
        # update probability
        uk = LH*np.dot(uk, TPM)
        uk /= np.sum(uk)
        # state union
        mode = np.argmax(uk)
        state = np.dot(uk, X)
        return mode, state, X, P, LH, uk

    def predict(self, measure, TPM, *args, **kwargs):
        mode, state, X, P, LH, uk = self.initial_state(measure[0])
        for k in range(1, self.frame):
            mode[k], state[k], X, P, LH, uk = self.IMMFilter(measure[k], TPM, X, P, LH, uk)
        return {f'{self.name} est_state': state, f'{self.name} est_mode': mode}


class ExactImmFilter(BasicFilter):
    pass


class NonAdaptiveImmFilter(BasicFilter):
    pass


class AIAdaptiveImmFilter(BasicFilter):
    def __init__(self, name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame, alpha):
        super().__init__(name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame)
        self.alpha = alpha

    def predict(self, measure, TPM, *args, **kwargs):
        knt = np.ones_like(TPM)*self.alpha
        his_TPM = np.zeros([self.frame, 3, 3])
        mode, state, X, P, LH, uk = self.initial_state(measure[0])
        for k in range(1, self.frame):
            mode[k], state[k], X, P, LH, uk = self.IMMFilter(measure[k], TPM, X, P, LH, uk)
            knt[mode[k-1], mode[k]] += 1
            TPM = knt/knt.sum(axis=1).reshape([-1, 1])
            his_TPM[k] = TPM
        return {f'{self.name} est_state': state, f'{self.name} est_mode': mode, f'{self.name} TPM': his_TPM}


class NIAdaptiveImmFilter(BasicFilter):
    def __init__(self, name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame, sticks):
        super().__init__(name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame)
        # initial the probability of each element of the numerical integration TPM
        self.sticks = sticks  # the dot nums splitted from the interval [0,1]
        self.grid = self.sticks*(self.sticks+1)//2
        self.NIProb = np.ones([self.grid, 3])/self.grid
        self.NITPA = np.empty([self.grid, 3])  # transition probability arrays
        row = 0
        for i in range(self.sticks):
            for j in range(self.sticks-i):
                self.NITPA[row] = i/(self.sticks-1), j/(self.sticks-1), 1-(i+j)/(self.sticks-1)
                row += 1

    def recursive_update_TPM(self, TPM, uk, LH, NIProb):
        """decoupled version of the numerical integration TPM estimator with N=231 grid vectors for each row"""
        # update NIProb
        eta = uk/np.linalg.multi_dot([uk, TPM, LH])
        for j in range(3):
            for i in range(self.grid):
                NIProb[i, j] *= (1+eta[j]*np.dot(self.NITPA[i]-TPM[j], LH))
        # update TPM
        TPM = np.dot(NIProb.T, self.NITPA)
        return TPM, NIProb

    def predict(self, measure, TPM, *args, **kwargs):
        his_TPM = np.zeros([self.frame, 3, 3])
        NIProb = self.NIProb.copy()
        mode, state, X, P, LH, uk = self.initial_state(measure[0])
        for k in range(1, self.frame):
            u_pre = uk.copy()
            mode[k], state[k], X, P, LH, uk = self.IMMFilter(measure[k], TPM, X, P, LH, uk)
            # update TPM
            TPM, NIProb = self.recursive_update_TPM(TPM, u_pre, LH, NIProb)
            his_TPM[k] = TPM
        return {f'{self.name} est_state': state, f'{self.name} est_mode': mode, f'{self.name} TPM': his_TPM}


class MyAdaptiveFilter(BasicFilter):
    def __init__(self, name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame, param):
        super().__init__(name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame)
        self.model = NNModel(param)

    def fit(self, X_train, y_train, is_valid=True, save_model=False, path=None):
        self.model.fit(X_train, y_train, is_valid=is_valid)
        if save_model:
            self.model.save_model(path)

    def predict(self, measure, TPM, *args, **kwargs):
        load_model = kwargs['load_model']
        path = kwargs['path']
        if load_model:
            self.model.load_model(path)
        sample = measure.shape[0]
        LH = np.empty([sample, 3], np.float32)
        uk = np.empty([sample, 3], np.float32)
        X = np.empty([sample, 3, 2], np.float32)
        P = np.empty([sample, 3, 2, 2], np.float32)
        mode = np.empty([sample, self.frame], np.float32)
        state = np.empty([sample, self.frame, 2], np.float32)
        his_TPM = np.empty([sample, self.frame, 3, 3], np.float32)
        for i in range(sample):
            mode[i], state[i], X[i], P[i], LH[i], uk[i] = self.initial_state(measure[i, 0])
        for k in range(1, self.frame):
            for i in range(sample):
                mode[i, k], state[i, k], X[i], P[i], LH[i], uk[i] = \
                    self.IMMFilter(measure[i, k], TPM[i], X[i], P[i], LH[i], uk[i])
            X_test = np.concatenate([np.zeros([sample, self.frame-1-k, 3], dtype=np.float32),
                                     np.array([mode[:, :k+1] == 0, mode[:, :k+1] == 1, mode[:, :k+1] == 2],
                                              dtype=np.float32).transpose([1, 2, 0])], axis=1).reshape([sample, -1])
            TPM = self.model.predict(X_test)
            his_TPM[:, k] = TPM
        return state, mode, his_TPM

