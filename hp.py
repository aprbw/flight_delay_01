import os
import sys
import time
import math
import random
import copy

import json

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import sklearn as sk

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from tensorflow.python.client import device_lib

# os.chdir('C:\\Users\\e91597\\Google Drive\\root RMIT\\CRUISE\\DL Flight Delay\\')
# os.chdir('/content/a/My Drive/root RMIT/CRUISE/DL Flight Delay/')
sys.path.append('./')
from Flight_Delay_experiment_5 import experiment
from FAA_DataPointLoader_Image_Only_2 import FAA_DataPointLoader_Image_Only

_fnw = './data/weather_201607_08_pre1.csv'
_fna = './data/Arrivals_TOTAL_withCallsign.csv'
_fnd = './data/Departures_TOTAL_withCallsign.csv'
_fnt = './data/unbreakable_traj_LAX_STAGE.csv'
_fn_dpls = './data/dpls.pickle'
_fn_dpli = './data/dpli_n28_unbrkbl.pickle'
_exp_name = 'minimal_run'
_fn_output = './outputs/flight_delay_output'
_frac_dataset = 1
_frac_val = 0.1
_frac_test = 0.2
_batch_size = 16
_l_cnn_layer = [8,8,8,8]
_l_fc_layer = [100]
_adam_lr = 10**-4
_n_early_stop_patience = 20
_n_epoch = 200
_ReduceLROnPlateau_monitor='val_loss'
_ReduceLROnPlateau_factor=0.1
_ReduceLROnPlateau_patience=10
_ReduceLROnPlateau_mode='min'
_ReduceLROnPlateau_min_delta=0
_ReduceLROnPlateau_cooldown=0
_ReduceLROnPlateau_min_lr=0
_observation_window_duration = 2400
_prediction_horizon = 14400
_c_weatherCondition = 'DROP'
_c_tailNumber = 'DROP'
_c_flightNumber = 'DROP'
_c_airport = 'DROP'
_c_time = 'UNIX'
_c_carrier_code = 'DROP'
_c_icao = 'DROP'
_c_callsign = 'DROP'
_normalization = 'MINMAX01'
_channel_duration=2400
_n_t_channel = 1
_nx=28
_ny=28
_count_filter = 'NONE'
_gauss_sigma = 0
_frac_sample = 0.000001
_verbose = True

while True:
    _exp_name = 'final_with_weather'
    _batch_size = 15
    _adam_lr = 3.48981*10**(-5)
    nf = 1
    n_cnn_layer = 2
    _l_cnn_layer = []
    _l_fc_layer = [429]
    _n_early_stop_patience = 20
    for i in range(n_cnn_layer):
        _l_cnn_layer.append(nf)
    try:
        experiment(
            fnw = _fnw,
            fna = _fna,
            fnd = _fnd,
            fnt = _fnt,
            fn_dpls = _fn_dpls,
            fn_dpli = _fn_dpli,
            exp_name = _exp_name,
            fn_output = _fn_output,
            frac_dataset = _frac_dataset,
            frac_val = _frac_val,
            frac_test = _frac_test,
            batch_size = _batch_size,
            l_cnn_layer = _l_cnn_layer,
            l_fc_layer = _l_fc_layer,
            adam_lr = _adam_lr,
            n_early_stop_patience = _n_early_stop_patience,
            n_epoch = _n_epoch,
            ReduceLROnPlateau_monitor = _ReduceLROnPlateau_monitor,
            ReduceLROnPlateau_factor = _ReduceLROnPlateau_factor,
            ReduceLROnPlateau_patience = _ReduceLROnPlateau_patience,
            ReduceLROnPlateau_mode = _ReduceLROnPlateau_mode,
            ReduceLROnPlateau_min_delta = _ReduceLROnPlateau_min_delta,
            ReduceLROnPlateau_cooldown = _ReduceLROnPlateau_cooldown,
            ReduceLROnPlateau_min_lr = _ReduceLROnPlateau_min_lr,
            observation_window_duration = _observation_window_duration,
            prediction_horizon = _prediction_horizon,
            c_weatherCondition = _c_weatherCondition,
            c_tailNumber = _c_tailNumber,
            c_flightNumber = _c_flightNumber,
            c_airport = _c_airport,
            c_time = _c_time,
            c_carrier_code = _c_carrier_code,
            c_icao = _c_icao,
            c_callsign = _c_callsign,
            normalization = _normalization,
            channel_duration = _channel_duration,
            n_t_channel = _n_t_channel,
            nx = _nx,
            ny = _ny,
            count_filter = _count_filter,
            gauss_sigma = _gauss_sigma,
            frac_sample = _frac_sample,
            verbose = _verbose)
    except:
        print(sys.exc_info()[0])



