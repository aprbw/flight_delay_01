import os
import sys
import argparse
import time
import json
import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from tensorflow.python.client import device_lib

sys.path.append('./')
from FAA_DataPointLoader_Image_Only_2 import FAA_DataPointLoader_Image_Only
from FAA_DataPointLoader_no_image_1 import FAA_DataPointLoader_no_image
from FAA_Sequence_1 import FAA_Sequence

_fnw = './data/weather_201607_08_pre1.csv'
_fna = './data/Arrivals_TOTAL_withCallsign.csv'
_fnd = './data/Departures_TOTAL_withCallsign.csv'
_fnt = './data/unbreakable_traj_LAX_STAGE.csv'
_fn_dpls = './data/dpls.pickle'
_fn_dpli = './data/dpli_n28_unbrnkbl_frac10m6.pickle'
_exp_name = 'minimal_run'
_fn_output = './outputs/flight_delay_output'
_frac_dataset = 1
_frac_val = 0.1
_frac_test = 0.2
_batch_size = 16
_l_cnn_layer = [3]
_l_fc_layer = [64]
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


def experiment(
               fnw = _fnw,
               fna = _fna,
               fnd = _fnd,
               fnt = _fnt,
               fn_dpli = _fn_dpli,
               fn_dpls = _fn_dpls,
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
               verbose = _verbose):
  # INIT
  args = locals()
  time_start = time.time()
  od = {} # output dictionary
  od['log'] = ['# FLIGHT DELAY EXPERIMENT']
  od['time_start'] = time_start
  fn_output += str(int(time_start))+'_'+str(os.getpid())+'.json'
  od['fn_output'] = fn_output
  def log(m):
      od['log'].append(m)
      if verbose:
          print(m)
      with open(fn_output, 'w') as outfile:
          json.dump(od, outfile)
  log((' == START == '))
  log(('time_start',time_start))
  log(('exp_name',exp_name))
  log(('fn_output',fn_output))
  od.update(args)
  log(('input arguments',args))
  if n_t_channel > 1:
      temporal_stride = (observation_window_duration - channel_duration) / (n_t_channel - 1)
  else:
      temporal_stride = 0
  ReduceLROnPlateau_verbose = int(verbose)
  od['temporal_stride'] = temporal_stride
  log(('temporal_stride',temporal_stride))
  delay_min = -24.0
  delay_max = 1175.0
  log(('delay_min',delay_min))
  log(('delay_max',delay_max))
  log(('# check cuda'))
  od['tensorflow_gpus'] = K.tensorflow_backend._get_available_gpus()
  log(('GPU',K.tensorflow_backend._get_available_gpus()))
  log(('initialization done',time.time()-time_start))

  # LOAD depart_unixdt_sorted
  log(('# LOAD depart_unixdt_sorted'))
  depart = pd.read_csv(fnd,usecols=['Date (MM/DD/YYYY)','Scheduled departure time'])
  depart_time_str = depart['Date (MM/DD/YYYY)'] + ' '+ depart['Scheduled departure time']
  depart_unixdt = (pd.to_datetime(depart_time_str).values.astype(float)//10**9).astype(int)
  depart_unixdt_sorted = np.argsort(depart_unixdt)
  log(('load depart_unixdt_sorted done',time.time()-time_start))

  # SPLIT
  log(('# SPLIT'))
  depart_unixdt_sorted = depart_unixdt_sorted[:int(frac_dataset * depart_unixdt_sorted.shape[0])]
  # TIME SERIES SPLIT
  # i_trn = depart_unixdt_sorted[:-int((frac_val + frac_test) * depart_unixdt_sorted.shape[0])] 
  # i_val = depart_unixdt_sorted[ -int((frac_val + frac_test) * depart_unixdt_sorted.shape[0]):-int(frac_test * depart_unixdt_sorted.shape[0])]
  # i_tnv = depart_unixdt_sorted[                                                             :-int(frac_test * depart_unixdt_sorted.shape[0])]
  # i_tst = depart_unixdt_sorted[ -int(            frac_test  * depart_unixdt_sorted.shape[0]):]
  
  # RANDOM SPLIT
  i_all = np.arange(depart_unixdt_sorted.shape[0])
  i_tst = np.random.choice(i_all,int(frac_test*depart_unixdt_sorted.shape[0]),replace=False)
  i_tnv = np.array(list(set(i_all) - set(i_tst)))
  i_val = np.random.choice(i_tnv,int(frac_val*depart_unixdt_sorted.shape[0]),replace=False)
  i_trn = np.array(list(set(i_tnv) - set(i_val)))
  log(('i_trn.shape',i_trn.shape))
  log(('i_val.shape',i_val.shape))
  log(('i_tnv.shape',i_val.shape))
  log(('i_tst.shape',i_tst.shape))
  log(('split done',time.time()-time_start))

  # LOAD DATA
  log(('# LOAD DATA'))
  if fn_dpls is None:
      log(('fn_dpls is None: generate...'))
      dpls = FAA_DataPointLoader_no_image(fnd,fna,fnw,
                   observation_window_duration = observation_window_duration,
                   prediction_horizon = prediction_horizon,
                   c_weatherCondition = c_weatherCondition,
                   c_tailNumber = c_tailNumber,
                   c_flightNumber = c_flightNumber,
                   c_airport = c_airport,
                   c_time = c_time,
                   c_carrier_code = c_carrier_code,
                   c_icao = c_icao,
                   c_callsign = c_callsign,
                   normalization = normalization,
                   verbose = verbose)
  else:
      log(('dpls load from file: ',fn_dpls))
      with open(fn_dpls, 'rb') as handle:
          dpls = pickle.load(handle)
  if fn_dpli is None:
      log(('fn_dpli is None: generate...'))
      dpli = FAA_DataPointLoader_Image_Only(fnt,fnd,
                   channel_duration = channel_duration,
                   prediction_horizon = prediction_horizon,
                   n_t_channel = n_t_channel,
                   temporal_stride = temporal_stride,
                   nx = nx,
                   ny = ny,
                   count_filter = count_filter,
                   gauss_sigma = gauss_sigma,
                   frac_sample = frac_sample,
                   verbose=verbose)
  else:
      log(('dpli load from file: ',fn_dpli))
      with open(fn_dpli, 'rb') as handle:
          dpli = pickle.load(handle)
  log(('load data done',time.time()-time_start))

  # DATA GENERATOR
  log(('# DATA GENERATOR'))
  trn_data_gen = FAA_Sequence(dpli,dpls,
       i_trn,
       batch_size = batch_size, is_shuffle=True,return_label = True)
  val_data_gen = FAA_Sequence(dpli,dpls,
       i_val,
       batch_size = batch_size, is_shuffle=False,return_label = True)
  tnv_data_gen = FAA_Sequence(dpli,dpls,
       i_tnv,
       batch_size = batch_size, is_shuffle=True,return_label = True)
  tst_data_gen = FAA_Sequence(dpli,dpls,
       i_tst,
       batch_size = batch_size, is_shuffle=False,return_label = True)
  log(('trn_data_gen[i_trn].Y.shape',trn_data_gen.dpls.Y[i_trn,:].shape))
  log(('val_data_gen[i_val].Y.shape',val_data_gen.dpls.Y[i_val,:].shape))
  log(('tst_data_gen[i_tst].Y.shape',tst_data_gen.dpls.Y[i_tst,:].shape))
  log(('data generator',time.time()-time_start))

  # MODEL
  log(('# MODEL'))
  def get_model():
      img_input_shape = dpli.getImageShape()
      s_input_shape = dpls.getFeatureShape()[1]
      output_shape = dpls.getLabelShape()[1]
      img_input = keras.layers.Input(shape=img_input_shape, name='img_input')
      temp_img = img_input
      for i_cnn in l_cnn_layer:
          temp_img = keras.layers.Conv2D(i_cnn, (3,3),padding='same',activation='relu')(temp_img)
          temp_img = keras.layers.MaxPooling2D(pool_size=(2, 2))(temp_img)
      img_flat = keras.layers.Flatten()(temp_img)
      s_input = keras.layers.Input(shape=(s_input_shape,), name='s_input')
      merged_result = keras.layers.concatenate([img_flat, s_input])
      temp_fc = merged_result
      for i_fc in l_fc_layer:
          temp_fc = keras.layers.Dense(i_fc, activation='relu')(temp_fc)
      output = keras.layers.Dense(output_shape, activation='linear')(temp_fc)
      model = keras.models.Model(inputs=[s_input,img_input], outputs=output)
      # optimizer
      optimizerAdam = keras.optimizers.Adam(lr=adam_lr)
      # compile
      model.compile(loss='mse', optimizer=optimizerAdam, metrics=['mse', 'mae', 'mape', 'cosine'])
      return model
  model = get_model()

  if verbose:
      model.summary(print_fn=log)
  log(('model',time.time()-time_start))

  # CALLBACKS
  log(('# CALLBACKS fit'))
  fn_model_best = './models/model_best'+str(int(time_start))+'_'+str(os.getpid())+'.h5'
  mcp_save = keras.callbacks.ModelCheckpoint(fn_model_best,
                                             save_best_only=True,
                                             monitor='val_loss',
                                             mode='min')
  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=n_early_stop_patience,
                                             verbose=int(verbose),
                                             mode='auto')
  reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor=ReduceLROnPlateau_monitor,
                                                        factor=ReduceLROnPlateau_factor,
                                                        patience=ReduceLROnPlateau_patience,
                                                        mode=ReduceLROnPlateau_mode,
                                                        min_delta=ReduceLROnPlateau_min_delta,
                                                        cooldown=ReduceLROnPlateau_cooldown,
                                                        min_lr=ReduceLROnPlateau_min_lr,
                                                        verbose=int(verbose))
  l_callbacks = [mcp_save,early_stop,reduce_on_plateau]
  log(('callbacks fit',time.time()-time_start))

  # FIT
  log(('# FIT'))
  model.fit_generator(trn_data_gen,
#                             steps_per_epoch=None,
                      epochs=n_epoch,
                      verbose=int(verbose),
                      callbacks=l_callbacks,
                      validation_data=val_data_gen,
#                             validation_steps=None,
                      # validation_freq=1,
                          # class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0)
  log(('end fit'))
  # model.history.history is a dict of numpy, and numpy cannot be json-ed
  d_model_trn_history = model.history.history.copy()
  for ik in d_model_trn_history.keys():
      for i2 in range(len(d_model_trn_history[ik])):
          d_model_trn_history[ik][i2] = float(d_model_trn_history[ik][i2])
  od['d_model_trn_history'] = d_model_trn_history
  log(('d_model_trn_history saved'))
  od['n_actual_epoch'] = len(d_model_trn_history[ik])
  log(('n_actual_epoch', len(d_model_trn_history[ik])))
  log(('fit',time.time()-time_start))

  log(('# VALIDATE'))
  model_best = keras.models.load_model(fn_model_best)
  valH_ = model_best.predict_generator(val_data_gen, verbose = int(verbose))[:,-1]
  valH = valH_ * (delay_max - delay_min) - delay_min
  valY = val_data_gen.dpls.Y[i_val,-1] * (delay_max - delay_min) - delay_min
  log(('valH.shape',valH.shape))
  val_mean_squared_error = sk.metrics.mean_squared_error(valY,valH)
  val_explained_variance_score = sk.metrics.explained_variance_score(valY,valH)
  val_mean_absolute_error = sk.metrics.mean_absolute_error(valY,valH)
  val_r2_score = sk.metrics.r2_score(valY,valH)
  od['val_mean_squared_error'] = val_mean_squared_error
  od['val_explained_variance_score'] = val_explained_variance_score
  od['val_mean_absolute_error'] = val_mean_absolute_error
  od['val_r2_score'] = val_r2_score
  log(('val_mean_squared_error',val_mean_squared_error))
  log(('val_explained_variance_score',val_explained_variance_score))
  log(('val_mean_absolute_error',val_mean_absolute_error))
  log(('val_r2_score',val_r2_score))
  log(('validate',time.time()-time_start))

  # CALLBACKS
  log(('# CALLBACKS tnv'))
  fn_model_best = './models/model_best'+str(int(time_start))+'_'+str(os.getpid())+'.h5'
  mcp_save = keras.callbacks.ModelCheckpoint(fn_model_best,
                                             save_best_only=True,
                                             monitor='loss',
                                             mode='min')
  early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                             min_delta=0,
                                             patience=n_early_stop_patience,
                                             verbose=int(verbose),
                                             mode='auto')
  reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                        factor=ReduceLROnPlateau_factor,
                                                        patience=ReduceLROnPlateau_patience,
                                                        mode=ReduceLROnPlateau_mode,
                                                        min_delta=ReduceLROnPlateau_min_delta,
                                                        cooldown=ReduceLROnPlateau_cooldown,
                                                        min_lr=ReduceLROnPlateau_min_lr,
                                                        verbose=int(verbose))
  l_callbacks = [mcp_save,early_stop,reduce_on_plateau]
  log(('callbacks tnv',time.time()-time_start))

  log(('# TNV'))
  model = get_model()
  model.fit_generator(tnv_data_gen,
#                             steps_per_epoch=None,
                      epochs=od['n_actual_epoch'] - n_early_stop_patience,
                      verbose=int(verbose),
                      callbacks=l_callbacks,
                      # validation_data=val_data_gen,
#                             validation_steps=None,
                      # validation_freq=1,
                          # class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0)
  log(('end tnv'))
  # model.history.history is a dict of numpy, and numpy cannot be json-ed
  d_model_tnv_history = model.history.history.copy()
  for ik in d_model_tnv_history.keys():
      for i2 in range(len(d_model_tnv_history[ik])):
          d_model_tnv_history[ik][i2] = float(d_model_tnv_history[ik][i2])
  od['d_model_tnv_history'] = d_model_tnv_history
  log(('d_model_tnv_history saved'))
  log(('tnv',time.time()-time_start))

  log(('# TEST'))
  model_best = keras.models.load_model(fn_model_best)
  tstH_ = model_best.predict_generator(tst_data_gen, verbose = int(verbose))[:,-1]
  tstH = tstH_ * (delay_max - delay_min) - delay_min
  tstY = tst_data_gen.dpls.Y[i_tst,-1] * (delay_max - delay_min) - delay_min
  log(('tstH.shape',tstH.shape))
  tst_mean_squared_error = sk.metrics.mean_squared_error(tstY,tstH)
  tst_explained_variance_score = sk.metrics.explained_variance_score(tstY,tstH)
  tst_mean_absolute_error = sk.metrics.mean_absolute_error(tstY,tstH)
  tst_r2_score = sk.metrics.r2_score(tstY,tstH)
  od['tst_mean_squared_error'] = tst_mean_squared_error
  od['tst_explained_variance_score'] = tst_explained_variance_score
  od['tst_mean_absolute_error'] = tst_mean_absolute_error
  od['tst_r2_score'] = tst_r2_score
  log(('tst_mean_squared_error',tst_mean_squared_error))
  log(('tst_explained_variance_score',tst_explained_variance_score))
  log(('tst_mean_absolute_error',tst_mean_absolute_error))
  log(('tst_r2_score',tst_r2_score))
  log(('test',time.time()-time_start))

  # save output
  od['time_end'] = time.time()
  log(('save time_end'))
  od['script_duration'] = od['time_end'] - time_start
  log(('save output',time.time()-time_start))
  print('fn_output',fn_output)
  log((' == FIN == '))
  return od

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fnw',default=_fnw,type=str)
    parser.add_argument('--fna',default=_fna,type=str)
    parser.add_argument('--fnd',default=_fnd,type=str)
    parser.add_argument('--fnt',default=_fnt,type=str)
    parser.add_argument('--fn_dpli',default=_fn_dpli,type=str)
    parser.add_argument('--fn_dpls',default=_fn_dpls,type=str)
    parser.add_argument('--exp_name',default=_exp_name,type=str)
    parser.add_argument('--fn_output',default=_fn_output,type=str)
    parser.add_argument('--frac_dataset',default=_frac_dataset,type=float)
    parser.add_argument('--frac_val',default=_frac_val,type=float)
    parser.add_argument('--frac_test',default=_frac_test,type=float)
    parser.add_argument('--batch_size',default=_batch_size,type=int)
    parser.add_argument('--l_cnn_layer',default=_l_cnn_layer,type=str)
    parser.add_argument('--l_fc_layer',default=_l_fc_layer,type=str)
    parser.add_argument('--adam_lr',default=_adam_lr,type=float)
    parser.add_argument('--n_early_stop_patience',default=_n_early_stop_patience,type=int)
    parser.add_argument('--n_epoch',default=_n_epoch,type=int)
    parser.add_argument('--ReduceLROnPlateau_monitor',default=_ReduceLROnPlateau_monitor,type=str)
    parser.add_argument('--ReduceLROnPlateau_factor',default=_ReduceLROnPlateau_factor,type=float)
    parser.add_argument('--ReduceLROnPlateau_patience',default=_ReduceLROnPlateau_patience,type=int)
    parser.add_argument('--ReduceLROnPlateau_mode',default=_ReduceLROnPlateau_mode,type=str)
    parser.add_argument('--ReduceLROnPlateau_min_delta',default=_ReduceLROnPlateau_min_delta,type=float)
    parser.add_argument('--ReduceLROnPlateau_cooldown',default=_ReduceLROnPlateau_cooldown,type=int)
    parser.add_argument('--ReduceLROnPlateau_min_lr',default=_ReduceLROnPlateau_min_lr,type=float)
    parser.add_argument('--observation_window_duration',default=_observation_window_duration,type=int)
    parser.add_argument('--prediction_horizon',default=_prediction_horizon,type=int)
    parser.add_argument('--c_weatherCondition',default=_c_weatherCondition,type=str)
    parser.add_argument('--c_tailNumber',default=_c_tailNumber,type=str)
    parser.add_argument('--c_flightNumber',default=_c_flightNumber,type=str)
    parser.add_argument('--c_airport',default=_c_airport,type=str)
    parser.add_argument('--c_time',default=_c_time,type=str)
    parser.add_argument('--c_carrier_code',default=_c_carrier_code,type=str)
    parser.add_argument('--c_icao',default=_c_icao,type=str)
    parser.add_argument('--c_callsign',default=_c_callsign,type=str)
    parser.add_argument('--normalization',default=_normalization,type=str)
    parser.add_argument('--channel_duration',default=_channel_duration,type=int)
    parser.add_argument('--n_t_channel',default=_n_t_channel,type=int)
    parser.add_argument('--nx',default=_nx,type=int)
    parser.add_argument('--ny',default=_ny,type=int)
    parser.add_argument('--count_filter',default=_count_filter,type=str)
    parser.add_argument('--gauss_sigma',default=_gauss_sigma,type=float)
    parser.add_argument('--frac_sample',default=_frac_sample,type=float)
    parser.add_argument('--verbose',default=_verbose,type=bool)
    args = parser.parse_args()  
    experiment(**vars(args)) 