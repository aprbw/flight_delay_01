import multiprocessing as mp
import numpy as np
import pandas as pd
import argparse
from scipy.ndimage.filters import gaussian_filter
import time
import sys
import os
import pickle
sys.path.append('./')
from ProgressIndicator import ProgressIndicator as API

_prediction_horizon = 4 * 60 * 60
_channel_duration=2*3600
_n_t_channel=3
_temporal_stride=3600
_nx=2
_ny=2
_count_filter='NONE' # NONE LOG BIN
_gauss_sigma=2
_frac_sample=0.001
_fn_dpli=None
_verbose=True

api = API(39391)

class FAA_DataPointLoader_Image_Only:
    def __init__(self,fn,fn_depart,
                 channel_duration=_channel_duration,
                 prediction_horizon=_prediction_horizon,
                 n_t_channel=_n_t_channel,
                 temporal_stride=_temporal_stride,
                 nx=_nx,
                 ny=_ny,
                 count_filter=_count_filter,
                 gauss_sigma=_gauss_sigma,
                 frac_sample=_frac_sample,
                 fn_dpli=_fn_dpli,
                 verbose=_verbose):
        '''
        channel_duration = duration per channel (i.e. conv filter size)
        this function does not take prediction_horizon into account
        '''
        ts = time.time()
        self.fn = fn
        self.channel_duration = channel_duration
        self.prediction_horizon = prediction_horizon
        self.n_t_channel = n_t_channel
        self.temporal_stride = temporal_stride
        self.nx = nx
        self.ny = ny
        self.count_filter = count_filter
        self.gauss_sigma = gauss_sigma
        self.frac_sample = frac_sample
        self.verbose = verbose
        # self.pakai_kolom_ini = ['time', 'latitude', 'longitude', 'altitude', 'heading', 'speed']
        self.pakai_kolom_ini = ['time', 'latitude', 'longitude', 'altitude', 'heading', 'speed']
        if mp.cpu_count()<3:
            self.n_cpu = mp.cpu_count()
        elif mp.cpu_count()<5:
            self.n_cpu = mp.cpu_count()-1
        else:
            self.n_cpu = mp.cpu_count()-2
        if verbose:
            print('fn',self.fn)
            print('channel_duration',self.channel_duration)
            print('n_t_channel',self.n_t_channel)
            print('temporal_stride',self.temporal_stride)
            print('nx',self.nx)
            print('ny',self.ny)
            print('count_filter',self.count_filter)
            print('gauss_sigma',self.gauss_sigma)
            print('frac_sample',self.frac_sample)
            print('verbose',self.verbose)
            print('pakai_kolom_ini',self.pakai_kolom_ini)
            print('n_cpu',self.n_cpu)
            print('n_cpu',self.n_cpu)
        if verbose:
            print('constant(ish) declared',time.time()-ts)
            
        depart = pd.read_csv(fn_depart,usecols=['Date (MM/DD/YYYY)','Scheduled departure time'])
        depart_time_str = depart['Date (MM/DD/YYYY)'] + ' '+ depart['Scheduled departure time']
        self.depart_unixdt = (pd.to_datetime(depart_time_str).values.astype(float)//10**9).astype(int)
        if verbose:
            print('depart_unixdt extracted',time.time()-ts)
            
        self.img_count = np.empty((self.depart_unixdt.shape[0],nx,ny,n_t_channel))
        self.img_speed = np.empty((self.depart_unixdt.shape[0],nx,ny,n_t_channel*2))
        if verbose:
            print('init done',time.time()-ts)
        df = pd.read_csv(fn,delimiter=',',index_col=0
                 ,usecols=self.pakai_kolom_ini)
        # df = pd.read_csv(fn,delimiter='|',index_col=0
        #                  ,usecols=self.pakai_kolom_ini)
        if verbose:
            print('load done',time.time()-ts)
            
        df = df.sample(frac = frac_sample)
        # processing
        df = df.loc[df['altitude']==0]
        df = df.drop(columns='altitude').reset_index()
        df['time'] = (pd.to_datetime(df['time']).values.astype(float)//10**9).astype(int)
        df['vx'] = df['speed'] * np.cos(df['heading'])
        df['vy'] = df['speed'] * np.sin(df['heading'])
        df.drop('speed',axis=1,inplace=True)
        df.drop('heading',axis=1,inplace=True)
        self.tmin = df['time'].min()
        self.t_start = self.tmin + channel_duration
        self.tmax = df['time'].max()
        self.t_end = self.tmax
        self.xmin = df['longitude'].min()
        self.xmax = df['longitude'].max()
        self.ymin = df['latitude'].min()
        self.ymax = df['latitude'].max()
        self.df = df
        if verbose:
            print('tmin',self.tmin)
            print('t_start',self.t_start)
            print('tmax',self.tmax)
            print('t_end',self.t_end)
            print('xmin',self.xmin)
            print('xmax',self.xmax)
            print('ymin',self.ymin)
            print('ymax',self.ymax)
            print('cleaning done',time.time()-ts)
        
        pool = mp.Pool(processes=self.n_cpu)
        r=[]
        global api
        if verbose:
            api.reinit(self.depart_unixdt.shape[0])
        else:
            api = None
        r = pool.starmap_async(
            parallel_assignment,
            [(i,self.depart_unixdt[i],self.get_count_channels,self.verbose) for i in range(len(self.depart_unixdt))]
            ).get()
        pool.close()
        pool.join()
        for ii,iv in r:
            self.img_count[ii,:,:,:] = iv
        if verbose:
            print('count pre-loaded done',time.time()-ts)
        
        if self.count_filter == 'LOG':
            self.img_count = np.log(self.img_count)
            self.img_count[self.img_count == -np.inf] = 0
        elif self.count_filter == 'BINARY':
            self.count_filter = np.where(self.count_filter>0,1.0,0.0) 
        self.count_ch_pixel_value_max = self.img_count.max()
        self.img_count /= self.count_ch_pixel_value_max
        if verbose:
            print('count pre loaded and normalized',time.time()-ts)
            
        pool = mp.Pool(processes=self.n_cpu)
        r=[]
        if verbose:
            api.reinit(self.depart_unixdt.shape[0])
        else:
            api = None
        r = pool.starmap_async(
            parallel_assignment,
            [(i,self.depart_unixdt[i],self.get_speed_channels,self.verbose) for i in range(len(self.depart_unixdt))]
            ).get()
        pool.close()
        pool.join()
        for ii,iv in r:
            self.img_speed[ii,:,:,:] = iv
        if verbose:
            print('speed pre-loaded done',time.time()-ts)

        self.speed_ch_pixel_value_max = max(abs(self.img_speed.min()),self.img_speed.max())
        self.img_speed /= self.speed_ch_pixel_value_max
        if verbose:
            print('speed pre loaded and normalized',time.time()-ts)

        if fn_dpli is not None:
            with open(fn_dpli, 'wb') as handle:
                pickle.dump(self, handle)
            if verbose:
                print('saved to:',fn_dpli)
        else:
            if verbose:
                print('dpli not saved')

        if verbose:
            print('FAA_DataPointLoader_Image_Only.__init__() FIN',time.time()-ts)
    def get_one_count_channel(self,key):
        '''
        key is unix time, not shifted ,not normalized
        '''
        tdf = self.df.loc[(key-self.channel_duration-self.prediction_horizon < self.df['time']) & (self.df['time'] < key-self.prediction_horizon)]
        try:
            img_1ch,_,_ = np.histogram2d(tdf['longitude'].values,tdf['latitude'].values,
                                         bins=[self.nx,self.ny],
                                         range=[[self.xmin,self.xmax],[self.ymin,self.ymax]])
        except:
            print(tdf)
            print(tdf['longitude'].values)
            print(tdf['latitude'].values)
        img_1ch = gaussian_filter(img_1ch,self.gauss_sigma,mode='constant')
        return img_1ch
    def get_one_pair_speed_velocity_channel(self,key):
        tdf = self.df.loc[ (key-self.channel_duration<self.df['time']) & (self.df['time']<key) ]
        vx,_,_ = np.histogram2d(tdf['longitude'].values,tdf['latitude'].values,
                                     weights = tdf['vx'],
                                     bins=[self.nx,self.ny],
                                     range=[[self.xmin,self.xmax],[self.ymin,self.ymax]])
        vy,_,_ = np.histogram2d(tdf['longitude'].values,tdf['latitude'].values,
                                     weights = tdf['vy'],
                                     bins=[self.nx,self.ny],
                                     range=[[self.xmin,self.xmax],[self.ymin,self.ymax]])
        vx = gaussian_filter(vx,self.gauss_sigma,mode='constant')
        vy = gaussian_filter(vy,self.gauss_sigma,mode='constant')
        return np.stack((vx,vy),axis=2)
    def get_count_channels(self,key):
        img_count = np.empty((self.nx,self.ny,self.n_t_channel))
        for ich in range(self.n_t_channel):
            img_count[:,:,ich] = self.get_one_count_channel(key-ich*self.temporal_stride)
        return img_count
    def get_speed_channels(self,key):
        img_speed = np.empty((self.nx,self.ny,self.n_t_channel*2))
        for ich in range(self.n_t_channel):
            img_speed[:,:,ich*2:2+ich*2] = self.get_one_pair_speed_velocity_channel(key-ich*self.temporal_stride)
        return img_speed
    def __getitem__(self,key):
        '''
        key is index of departure
        if you want the normal index, use:
        .img_count and .img_speed
        '''
        # key = np.where(self.depart_unixdt == key)[0][0]
        return np.concatenate((self.img_count[key,:,:,:],self.img_speed[key,:,:,:]),axis=2)
    def __len__(self):
        return self.t_end - self.t_start
    def getImageShape(self):
        return (self.nx,self.ny,3*self.n_t_channel)

def parallel_assignment(ii,i_depart_unixdt,dfl_fn,verbose):
    values_at_ii = dfl_fn(i_depart_unixdt)
    if verbose:
        global api
        api.toc('pre-loading',ii)
    return ii,values_at_ii

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn',type=str, default=str(os.path.join('data','unbreakable_traj_LAX_STAGE.csv')))
    parser.add_argument('--fn_depart',type=str, default=str(os.path.join('data','Departures_TOTAL_withCallsign.csv')))
    parser.add_argument('--channel_duration',type=int, default=_channel_duration)
    parser.add_argument('--prediction_horizon',type=int, default=_prediction_horizon)
    parser.add_argument('--n_t_channel',type=int, default=_n_t_channel)
    parser.add_argument('--temporal_stride',type=int, default=_temporal_stride)
    parser.add_argument('--nx',type=int, default=_nx)
    parser.add_argument('--ny',type=int, default=_ny)
    parser.add_argument('--count_filter',type=str, default=_count_filter)
    parser.add_argument('--gauss_sigma',type=int, default=_gauss_sigma)
    parser.add_argument('--frac_sample',type=float, default=_frac_sample)
    parser.add_argument('--fn_dpli',type=str, default=_fn_dpli)
    parser.add_argument('--verbose',type=bool, default=_verbose)
    args = parser.parse_args()
    dpli = FAA_DataPointLoader_Image_Only(**vars(args))
