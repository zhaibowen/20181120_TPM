# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 19:16
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : main.py
"""

"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import timer, frame_expand, parallel_predict
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from track_generator import track_generator
from config import acc, trans, noits, meats, u0, p0_prob, v0_prob, wk_prob, vk_prob, total_frame, processor, \
    Exact_name, NonAdaptive_name, NIAdaptive_name, MyAdaptive_name, NonAdaptive_TPM, prior_TPM, NIAdaptive_sticks, \
    train_size, test_size, save_model, is_valid, nn_param, track_data_train_file, track_data_test_file, \
    test_result_file, scaler_measure_file, model_file, AIAdaptive_name, AI_prior_alpha
from Filters import ExactImmFilter, NonAdaptiveImmFilter, NIAdaptiveImmFilter, MyAdaptiveFilter, AIAdaptiveImmFilter
from Metrics import position_MAE, velocity_MAE, TPM_MAE, Mean_true_mode_probability


def main():
    # generate and save tracks
    # with timer(f'track generate: sample {train_size+test_size}, frame {total_frame}'):
    #     train_data = track_generator(train_size, total_frame, acc, trans, noits, u0, p0_prob, v0_prob, wk_prob, vk_prob)
    #     pickle.dump(train_data, open(track_data_train_file, 'wb'))
    #     test_data = track_generator(test_size, total_frame, acc, trans, noits, u0, p0_prob, v0_prob, wk_prob, vk_prob)
    #     pickle.dump(test_data, open(track_data_test_file, 'wb'))
    #
    # # load test_data, TPM, modes, tracks, measurements
    # test_data = pickle.load(open(track_data_test_file, 'rb'))
    #
    # # ExactImmFilter
    # model = ExactImmFilter(Exact_name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame)
    # with Pool(processes=processor) as pool, timer(f'{Exact_name} predict'):
    #     test_data = pool.map(partial(parallel_predict, func=model.predict, TPM=Exact_name), test_data)
    #
    # # NonAdaptiveImmFilter
    # model = NonAdaptiveImmFilter(NonAdaptive_name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame)
    # with Pool(processes=processor) as pool, timer(f'{NonAdaptive_name} predict'):
    #     test_data = pool.map(partial(parallel_predict, func=model.predict, TPM=NonAdaptive_TPM), test_data)
    #
    # # NIAdaptiveImmFilter
    # model = NIAdaptiveImmFilter(NIAdaptive_name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob,
    #                             total_frame, NIAdaptive_sticks)
    # with Pool(processes=processor) as pool, timer(f'{NIAdaptive_name} predict'):
    #     test_data = pool.map(partial(parallel_predict, func=model.predict, TPM=prior_TPM), test_data)
    #
    # # AIAdaptiveImmFilter
    # model = AIAdaptiveImmFilter(AIAdaptive_name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame, AI_prior_alpha)
    # with Pool(processes=processor) as pool, timer(f'{AIAdaptive_name} predict'):
    #     test_data = pool.map(partial(parallel_predict, func=model.predict, TPM=prior_TPM), test_data)
    # pickle.dump(test_data, open(test_result_file, 'wb'))
    #
    # NNAdaptiveFilter
    model = MyAdaptiveFilter(MyAdaptive_name, acc, trans, noits, meats, u0, v0_prob, wk_prob, vk_prob, total_frame, nn_param)

    # NNAdaptiveFilter train model
    with timer(f'{MyAdaptive_name} fit'):
        train_data = pickle.load(open(track_data_train_file, 'rb'))
        mode = np.array(list(map(lambda x: x['mode'], train_data)), dtype=np.float32)
        X_train = frame_expand(np.array([mode == 0, mode == 1, mode == 2], dtype=np.float32).transpose([1, 2, 0]))
        y_train = np.array(list(map(lambda x: x['TPM'], train_data)))
        y_train = np.tile(y_train, [total_frame-1, 1, 1, 1]).transpose([1, 0, 2, 3]).\
            reshape([train_size*(total_frame-1), 3, 3])
        model.fit(X_train, y_train, is_valid=is_valid, save_model=save_model, path=model_file)
    #
    # # NNAdaptiveFilter predict
    # with timer(f'{MyAdaptive_name} predict'):
    #     measure = np.array(list(map(lambda x: x['measure'], test_data)))
    #     TPM = np.tile(prior_TPM, [len(test_data), 1, 1])
    #     state, mode, his_TPM = model.predict(measure, TPM, load_model=save_model, path=model_file)
    #     for i in range(len(test_data)):
    #         test_data[i].update({f'{model.name} est_state': state[i], f'{model.name} est_mode': mode[i],
    #                              f'{model.name} TPM': his_TPM[i]})

    # save result of ExactImmFilter, NonAdaptiveImmFilter, NIAdaptiveImmFilter, NNAdaptiveFilter
    # pickle.dump(test_data, open(test_result_file, 'wb'))
    # load test_data
    test_data = pickle.load(open(test_result_file, 'rb'))

    # Metrics
    position_MAE(test_data)
    velocity_MAE(test_data)
    Mean_true_mode_probability(test_data)
    TPM_MAE(test_data)
    plt.show()
    np.log

if __name__ == '__main__':
    main()
