from pprint import pprint
import pickle
import time
from datetime import datetime, timedelta
import json
import math
import random
import numpy as np
import pandas as pd
import sys
sys.path.append('./')
from ProgressIndicator import ProgressIndicator as API
import warnings
warnings.filterwarnings('ignore')

class FAA_DataPointLoader_no_image:
    # choose between DROP or ONEHOT
    # time, 'UNIX' or 'COMPLETE'
    # complete: also,month,day,day_of_week
    def __init__(self,fn_dep,fn_arr,_fn_wthr,
                 observation_window_duration,prediction_horizon,
                 c_weatherCondition = 'DROP', # 'ONEHOT'
                 c_tailNumber = 'DROP', # 'ONEHOT'
                 c_flightNumber = 'DROP', # ONEHOT
                 c_airport = 'DROP', # ONEHOT
                 c_time = 'UNIX', # COMPLETE (no year)
                 c_carrier_code = 'DROP', # ONEHOT
                 c_icao = 'DROP', # (always drop)
                 c_callsign = 'DROP', # (always drop)
                 normalization = 'MINMAX01', # NONE, STANDARD, (WHITE)
                 verbose = False):
      time_start = time.time()

      self.fn_dep = fn_dep
      self.fn_arr = fn_arr
      self._fn_wthr = _fn_wthr
      self.observation_window_duration = observation_window_duration
      self.prediction_horizon = prediction_horizon
      self.c_weatherCondition = c_weatherCondition
      self.c_tailNumber = c_tailNumber
      self.c_flightNumber = c_flightNumber
      self.c_airport = c_airport
      self.c_time = c_time
      self.c_carrier_code = c_carrier_code
      self.c_icao = c_icao
      self.c_callsign = c_callsign
      self.normalization = normalization
      self.verbose = verbose

      weather = pd.read_csv(_fn_wthr)   # data cleaning here
      arrival = pd.read_csv(fn_arr)
      depart = pd.read_csv(fn_dep)

      weather['Wind_is_CALM'] = weather.Wind
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("CALM", "1")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("WSW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("VAR", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("SSE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("NNW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("SSW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("ESE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("WNW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("ENE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("NNE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("SW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("SE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("NW", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("NE", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("W", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("E", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("S", "0")
      weather['Wind_is_CALM'] = weather.Wind_is_CALM.str.replace("N", "0")
      weather['Wind_is_CALM'] = weather['Wind_is_CALM'].astype(float)
      weather['Wind_is_VAR'] = weather.Wind
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("VAR", "1")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("WSW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("CALM", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("SSE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("NNW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("SSW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("ESE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("WNW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("ENE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("NNE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("SW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("SE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("NW", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("NE", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("W", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("E", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("S", "0")
      weather['Wind_is_VAR'] = weather.Wind_is_VAR.str.replace("N", "0")
      weather['Wind_is_VAR'] = weather['Wind_is_VAR'].astype(float)
      # convert "Wind" from direction to radian
      weather['Wind'] = weather.Wind.str.replace("WSW", str(247.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("CALM", str(0))
      weather['Wind'] = weather.Wind.str.replace("VAR", str(0))
      weather['Wind'] = weather.Wind.str.replace("SSE", str(157.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("NNW", str(337.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("SSW", str(202.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("ESE", str(112.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("WNW", str(292.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("ENE", str(67.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("NNE", str(22.5*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("SW", str(225*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("SE", str(135*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("NW", str(315*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("NE", str(45*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("W", str(270*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("E", str(90*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("S", str(180*math.pi/180))
      weather['Wind'] = weather.Wind.str.replace("N", str(0*math.pi/180))
      weather['Wind'] = weather['Wind'].astype(float)

      weather.DateTime = pd.to_datetime(weather.DateTime)
      if self.c_time == 'COMPLETE':
        weather['year'], weather['month'], weather['day_of_month'] = weather['DateTime'].dt.year, weather['DateTime'].dt.month, weather['DateTime'].dt.day
        #weather['year'] = (weather.year - weather.year.min()) / (weather.year.max() - weather.year.min())
        weather['Date'] = weather['month'].apply(str) + '/' + weather['day_of_month'].apply(str) + '/' + weather['year'].apply(str)
        #weather['Date'] = (weather.Date - weather.Date.min()) / (weather.Date.max() - weather.Date.min())
        weather['day_of_year'] = weather['DateTime'].dt.dayofyear
        #weather['day_of_year'] = (weather.day_of_year - weather.day_of_year.min()) / (weather.day_of_year.max() - weather.day_of_year.min())
        weather['day_of_week'] = weather['DateTime'].dt.dayofweek
        #weather['day_of_week'] = (weather.day_of_week - weather.day_of_week.min()) / (weather.day_of_week.max() - weather.day_of_week.min())
        weather['week_of_year'] = weather['DateTime'].dt.weekofyear
        #weather['week_of_year'] = (weather.week_of_year - weather.week_of_year.min()) / (weather.week_of_year.max() - weather.week_of_year.min())
        weather['hour_in_day'] = weather['DateTime'].dt.hour
        #weather['hour_in_day'] = (weather.hour_in_day - weather.hour_in_day.min()) / (weather.hour_in_day.max() - weather.hour_in_day.min())
        weather['minute_in_hour'] = weather['DateTime'].dt.minute
        #weather['minute_in_hour'] = (weather.minute_in_hour - weather.minute_in_hour.min()) / (weather.minute_in_hour.max() - weather.minute_in_hour.min())
        weather['time_in_unix'] = weather['DateTime'].apply(lambda x: datetime.timestamp(x))
      elif self.c_time == 'UNIX':
        weather['time_in_unix'] = weather['DateTime'].apply(lambda x: datetime.timestamp(x))
             
      if self.c_weatherCondition == 'ONEHOT':
        weather = weather.join(pd.get_dummies(weather.Condition))
      elif self.c_weatherCondition == 'DROP':
        # make it drop
        pass
      
      weather = weather.drop(['Date','Unnamed: 0','Condition','Time'], axis=1)

      weather['Temperature'] = (weather.Temperature - weather.Temperature.min()) / (weather.Temperature.max() - weather.Temperature.min())
      weather['DewPoint'] = (weather.DewPoint - weather.DewPoint.min()) / (weather.DewPoint.max() - weather.DewPoint.min())
      weather['Wind'] = (weather.Wind - weather.Wind.min()) / (weather.Wind.max() - weather.Wind.min())
      weather['WindSpeed'] = (weather.WindSpeed - weather.WindSpeed.min()) / (weather.WindSpeed.max() - weather.WindSpeed.min())
      weather['WindGust'] = (weather.WindGust - weather.WindGust.min()) / (weather.WindGust.max() - weather.WindGust.min())
      weather['Pressure'] = (weather.Pressure - weather.Pressure.min()) / (weather.Pressure.max() - weather.Pressure.min())
      weather['temp_unix'] = (weather.time_in_unix - weather.time_in_unix.min()) / (weather.time_in_unix.max() - weather.time_in_unix.min())

      self.weather = weather

      if verbose: print('weather DONE, START arrival',time.time()-time_start)

      # rename columns
      arrival.rename(columns={'Date (MM/DD/YYYY)':'Date', 'Tail Number':'Tail_Number', 'Scheduled Arrival Time':'Scheduled_Arrival_Time',
                              'Carrier Code':'Carrier_Code', 'Origin Airport':'Origin_Airport', 'Actual Arrival Time':'Actual_Arrival_Time',
                              'Wheels-on Time':'Wheels_on_Time', 'Scheduled Elapsed Time (Minutes)':'Scheduled_Elapsed_Time_Minutes',
                              'Actual Elapsed Time (Minutes)':'Actual_Elapsed_Time_Minutes', 'Arrival Delay (Minutes)':'Arrival_Delay_Minutes',
                              'Taxi-In time (Minutes)':'Taxi_In_time_Minutes', 'Delay Carrier (Minutes)':'Delay_Carrier_Minutes', 
                              'Delay Weather (Minutes)':'Delay_Weather_Minutes', 
                              'Delay National Aviation System (Minutes)':'Delay_National_Aviation_System_Minutes',
                              'Delay Security (Minutes)':'Delay_Security_Minutes', 
                              'Delay Late Aircraft Arrival (Minutes)':'Delay_Late_Aircraft_Arrival_Minutes'}, inplace = True)
     
      arrival['DateTime'] = arrival['Date'] + ' ' + arrival['Scheduled_Arrival_Time']
      arrival.DateTime = pd.to_datetime(arrival.DateTime)
      if self.c_time == 'COMPLETE':
        arrival['year'], arrival['month'] = arrival['DateTime'].dt.year, arrival['DateTime'].dt.month
        arrival['day_of_month'] = arrival['DateTime'].dt.day
        arrival['Date'] = arrival['month'].apply(str) + '/' + arrival['day_of_month'].apply(str) + '/' + arrival['year'].apply(str)
        arrival['day_of_year'], arrival['day_of_week'] = arrival['DateTime'].dt.dayofyear, arrival['DateTime'].dt.dayofweek
        arrival['week_of_year'] = arrival['DateTime'].dt.weekofyear
        arrival['hour_in_day_SAT'], arrival['minute_in_hour_SAT'] = arrival['DateTime'].dt.hour, arrival['DateTime'].dt.minute
        arrival['time_in_unix_SAT'] = arrival['DateTime'].apply(lambda x: datetime.timestamp(x))

        arrival['Actual_Arrival_Time'] = arrival.Actual_Arrival_Time.str.replace("24:00", "23:59")
        arrival['DateTime_AAT'] = arrival['Date'] + ' ' + arrival['Actual_Arrival_Time']
        arrival.DateTime_AAT = pd.to_datetime(arrival.DateTime_AAT)
        arrival['hour_in_day_AAT'], arrival['minute_in_hour_AAT'] = arrival['DateTime_AAT'].dt.hour, arrival['DateTime_AAT'].dt.minute
        arrival['time_in_unix_AAT'] = arrival['DateTime_AAT'].apply(lambda x: datetime.timestamp(x))

        arrival['Wheels_on_Time'] = arrival.Wheels_on_Time.str.replace("24:00", "23:59")
        arrival['DateTime_WOnT'] = arrival['Date'] + ' ' + arrival['Wheels_on_Time']
        arrival.DateTime_WOnT = pd.to_datetime(arrival.DateTime_WOnT)
        arrival['hour_in_day_WOnT'], arrival['minute_in_hour_WOnT'] = arrival['DateTime_WOnT'].dt.hour, arrival['DateTime_WOnT'].dt.minute
        arrival['time_in_unix_WOnT'] = arrival['DateTime_WOnT'].apply(lambda x: datetime.timestamp(x))
      elif self.c_time == 'UNIX':
        arrival['time_in_unix_SAT'] = arrival['DateTime'].apply(lambda x: datetime.timestamp(x))    
        #arrival['time_in_unix_SAT'] = pd.to_datetime(arrival.DateTime).values.astype(float)//10**9 
        arrival['Actual_Arrival_Time'] = arrival.Actual_Arrival_Time.str.replace("24:00", "23:59")
        arrival['DateTime_AAT'] = arrival['Date'] + ' ' + arrival['Actual_Arrival_Time']
        arrival.DateTime_AAT = pd.to_datetime(arrival.DateTime_AAT)        
        arrival['time_in_unix_AAT'] = arrival['DateTime_AAT'].apply(lambda x: datetime.timestamp(x))
        #arrival['time_in_unix_AAT'] = pd.to_datetime(arrival.DateTime_AAT).values.astype(float)//10**9
        arrival['Wheels_on_Time'] = arrival.Wheels_on_Time.str.replace("24:00", "23:59")
        arrival['DateTime_WOnT'] = arrival['Date'] + ' ' + arrival['Wheels_on_Time']
        arrival.DateTime_WOnT = pd.to_datetime(arrival.DateTime_WOnT)
        arrival['time_in_unix_WOnT'] = arrival['DateTime_WOnT'].apply(lambda x: datetime.timestamp(x))
        #arrival['time_in_unix_WOnT'] = pd.to_datetime(arrival.DateTime_WOnT).values.astype(float)//10**9

      if self.c_carrier_code == 'ONEHOT':
        arrival = arrival.join(pd.get_dummies(arrival.Carrier_Code))
      elif self.c_carrier_code == 'DROP':
        # make it drop
        pass
      if self.c_airport == 'ONEHOT':
        arrival = arrival.join(pd.get_dummies(arrival.Origin_Airport))
      elif self.c_airport == 'DROP':
        # make it drop
        pass
      if self.c_tailNumber == 'ONEHOT':
        arrival = arrival.join(pd.get_dummies(arrival.Tail_Number))
      elif self.c_tailNumber == 'DROP':
        # make it drop
        pass

      arrival = arrival.drop(['Flight Number','Date','Scheduled_Arrival_Time','Actual_Arrival_Time','Wheels_on_Time','DateTime_AAT','DateTime_WOnT',
                              'Carrier_Code', 'Origin_Airport','ICAO','Callsign'], axis=1)
      
      arrival['Scheduled_Elapsed_Time_Minutes'] = (arrival.Scheduled_Elapsed_Time_Minutes - arrival.Scheduled_Elapsed_Time_Minutes.min()) / (arrival.Scheduled_Elapsed_Time_Minutes.max() - arrival.Scheduled_Elapsed_Time_Minutes.min())
      arrival['Actual_Elapsed_Time_Minutes'] = (arrival.Actual_Elapsed_Time_Minutes - arrival.Actual_Elapsed_Time_Minutes.min()) / (arrival.Actual_Elapsed_Time_Minutes.max() - arrival.Actual_Elapsed_Time_Minutes.min())      
      arrival['Arrival_Delay_Minutes'] = (arrival.Arrival_Delay_Minutes - arrival.Arrival_Delay_Minutes.min()) / (arrival.Arrival_Delay_Minutes.max() - arrival.Arrival_Delay_Minutes.min())
      arrival['Taxi_In_time_Minutes'] = (arrival.Taxi_In_time_Minutes - arrival.Taxi_In_time_Minutes.min()) / (arrival.Taxi_In_time_Minutes.max() - arrival.Taxi_In_time_Minutes.min())
      arrival['Delay_Carrier_Minutes'] = (arrival.Delay_Carrier_Minutes - arrival.Delay_Carrier_Minutes.min()) / (arrival.Delay_Carrier_Minutes.max() - arrival.Delay_Carrier_Minutes.min())
      arrival['Delay_Weather_Minutes'] = (arrival.Delay_Weather_Minutes - arrival.Delay_Weather_Minutes.min()) / (arrival.Delay_Weather_Minutes.max() - arrival.Delay_Weather_Minutes.min())
      arrival['Delay_National_Aviation_System_Minutes'] = (arrival.Delay_National_Aviation_System_Minutes - arrival.Delay_National_Aviation_System_Minutes.min()) / (arrival.Delay_National_Aviation_System_Minutes.max() - arrival.Delay_National_Aviation_System_Minutes.min())
      arrival['Delay_Security_Minutes'] = (arrival.Delay_Security_Minutes - arrival.Delay_Security_Minutes.min()) / (arrival.Delay_Security_Minutes.max() - arrival.Delay_Security_Minutes.min())
      arrival['Delay_Late_Aircraft_Arrival_Minutes'] = (arrival.Delay_Late_Aircraft_Arrival_Minutes - arrival.Delay_Late_Aircraft_Arrival_Minutes.min()) / (arrival.Delay_Late_Aircraft_Arrival_Minutes.max() - arrival.Delay_Late_Aircraft_Arrival_Minutes.min())
      arrival['time_in_unix_SAT'] = (arrival.time_in_unix_SAT - arrival.time_in_unix_SAT.min()) / (arrival.time_in_unix_SAT.max() - arrival.time_in_unix_SAT.min())
      arrival['time_in_unix_WOnT'] = (arrival.time_in_unix_WOnT - arrival.time_in_unix_WOnT.min()) / (arrival.time_in_unix_WOnT.max() - arrival.time_in_unix_WOnT.min())
      arrival['temp_AAT'] = (arrival.time_in_unix_AAT - arrival.time_in_unix_AAT.min()) / (arrival.time_in_unix_AAT.max() - arrival.time_in_unix_AAT.min())

      self.arrival = arrival

      if verbose: print('arrival DONE, START depart',time.time()-time_start)

      # rename "Date (MM/DD/YYYY)" & "Tail Number" columns
      depart.rename(columns={'Date (MM/DD/YYYY)':'Date', 'Tail Number':'Tail_Number', 'Scheduled departure time':'Scheduled_departure_time',
                             'Actual departure time':'Actual_departure_time', 'Wheels-off time':'Wheels_off_time', 'Carrier Code':'Carrier_Code',
                             'Destination Airport':'Destination_Airport', 'Scheduled elapsed time (Minutes)':'Scheduled_elapsed_time_Minutes',
                             'Departure delay (Minutes)':'Departure_delay_Minutes', 'Delay Carrier (Minutes)':'Delay_Carrier_Minutes',
                             'Delay Weather (Minutes)':'Delay_Weather_Minutes', 'Delay National Aviation System (Minutes)':'Delay_National_Aviation_System_Minutes',
                             'Delay Security (Minutes)':'Delay_Security_Minutes', 'Delay Late Aircraft Arrival (Minutes)':'Delay_Late_Aircraft_Arrival_Minutes'}, inplace = True)

      # correct day and month and create a new column 'DateTime'
      depart['DateTime'] = depart['Date'] + ' ' + depart['Scheduled_departure_time']
      depart.DateTime = pd.to_datetime(depart.DateTime)
      depart['year'], depart['month'], depart['day_of_month'] = depart['DateTime'].dt.year, depart['DateTime'].dt.day, depart['DateTime'].dt.month
      depart['Date'] = depart['month'].apply(str) + '/' + depart['day_of_month'].apply(str) + '/' + depart['year'].apply(str)
      depart['DateTime'] = depart['Date'] + ' ' + depart['Scheduled_departure_time']
      depart.DateTime = pd.to_datetime(depart.DateTime)

      if self.c_time == 'COMPLETE':
        depart['day_of_year'], depart['day_of_week'] = depart['DateTime'].dt.dayofyear, depart['DateTime'].dt.dayofweek
        depart['week_of_year'] = depart['DateTime'].dt.weekofyear
        depart['hour_in_day_SDT'], depart['minute_in_hour_SDT'] = depart['DateTime'].dt.hour, depart['DateTime'].dt.minute
        depart['time_in_unix_SDT'] = depart['DateTime'].apply(lambda x: datetime.timestamp(x))
      elif self.c_time == 'UNIX':
        depart['time_in_unix_SDT'] = depart['DateTime'].apply(lambda x: datetime.timestamp(x))
        depart = depart.drop(['year','month','day_of_month'], axis=1)

      if self.c_carrier_code == 'ONEHOT':
        depart = depart.join(pd.get_dummies(depart.Carrier_Code))
      elif self.c_carrier_code == 'DROP':
        pass
      if self.c_airport == 'ONEHOT':
        depart = depart.join(pd.get_dummies(depart.Destination_Airport))
      elif self.c_airport == 'DROP':
        # make it drop
        pass
      if self.c_tailNumber == 'ONEHOT':
        depart = depart.join(pd.get_dummies(depart.Tail_Number))
      elif self.c_tailNumber == 'DROP':
        # make it drop
        pass

      depart = depart.drop(['Flight Number','Date','Scheduled_departure_time','Actual_departure_time','Wheels_off_time', 'Carrier_Code',
                            'Destination_Airport','ICAO','Callsign','Actual elapsed time (Minutes)','Taxi-Out time (Minutes)'], axis=1)
      
      depart['Scheduled_elapsed_time_Minutes'] = (depart.Scheduled_elapsed_time_Minutes - depart.Scheduled_elapsed_time_Minutes.min()) / (depart.Scheduled_elapsed_time_Minutes.max() - depart.Scheduled_elapsed_time_Minutes.min())
      depart['Departure_delay_Minutes'] = (depart.Departure_delay_Minutes - depart.Departure_delay_Minutes.min()) / (depart.Departure_delay_Minutes.max() - depart.Departure_delay_Minutes.min())
      depart['Delay_Carrier_Minutes'] = (depart.Delay_Carrier_Minutes - depart.Delay_Carrier_Minutes.min()) / (depart.Delay_Carrier_Minutes.max() - depart.Delay_Carrier_Minutes.min())
      depart['Delay_Weather_Minutes'] = (depart.Delay_Weather_Minutes - depart.Delay_Weather_Minutes.min()) / (depart.Delay_Weather_Minutes.max() - depart.Delay_Weather_Minutes.min())
      depart['Delay_National_Aviation_System_Minutes'] = (depart.Delay_National_Aviation_System_Minutes - depart.Delay_National_Aviation_System_Minutes.min()) / (depart.Delay_National_Aviation_System_Minutes.max() - depart.Delay_National_Aviation_System_Minutes.min())
      depart['Delay_Security_Minutes'] = (depart.Delay_Security_Minutes - depart.Delay_Security_Minutes.min()) / (depart.Delay_Security_Minutes.max() - depart.Delay_Security_Minutes.min())
      depart['Delay_Late_Aircraft_Arrival_Minutes'] = (depart.Delay_Late_Aircraft_Arrival_Minutes - depart.Delay_Late_Aircraft_Arrival_Minutes.min()) / (depart.Delay_Late_Aircraft_Arrival_Minutes.max() - depart.Delay_Late_Aircraft_Arrival_Minutes.min())
      depart['temp_SDT'] = (depart.time_in_unix_SDT - depart.time_in_unix_SDT.min()) / (depart.time_in_unix_SDT.max() - depart.time_in_unix_SDT.min())
      
      self.depart = depart

      if verbose: print('depart DONE, continue...',time.time()-time_start)

      if (self.c_time == 'UNIX' and self.c_carrier_code == 'DROP' and self.c_airport == 'DROP' and self.c_tailNumber == 'DROP' and self.c_weatherCondition == 'DROP'):
        self.x_col = 24
        self.y_col = 6
      else:
        raise NotImplementedError(self.c_time, self.c_carrier_code, self.c_airport, self.c_tailNumber, self.c_weatherCondition)
      tlen  = len(depart)
      self.X = np.empty((tlen,self.x_col))
      self.Y = np.empty((tlen,self.y_col))
      if verbose:
        api = API(tlen)
      for i in range(tlen):
        self.X[i,:],self.Y[i,:] = self.getItem_in_np(i)
        if verbose: api.toc('matching...')
      return

    def __len__(self):      # index available value
        length = len(self.depart)  
        return length

    def __getitem__(self,key):
      return self.X[key,:],self.Y[key,:]

    def getItem_in_df(self,indexNum):
      df_y = self.depart.query('index == @indexNum')
      tailNumb = df_y["Tail_Number"].values[0]
      matchTime = df_y["time_in_unix_SDT"].values[0]
      timePoint = matchTime - self.prediction_horizon

      label_y = df_y.loc[:,["Departure_delay_Minutes", "Delay_Carrier_Minutes", "Delay_Weather_Minutes",
                            "Delay_National_Aviation_System_Minutes", "Delay_Security_Minutes", "Delay_Late_Aircraft_Arrival_Minutes"]]

      x_dep_temp = df_y.drop(["Departure_delay_Minutes", "Delay_Carrier_Minutes", "Delay_Weather_Minutes",
                              "Delay_National_Aviation_System_Minutes", "Delay_Security_Minutes", "Delay_Late_Aircraft_Arrival_Minutes",
                              "Tail_Number","DateTime"], axis=1)  
      
      df_xarr = self.arrival.query('Tail_Number == @tailNumb & time_in_unix_AAT <= @matchTime').tail(1)
      x_arr_temp = df_xarr.drop(["Tail_Number","DateTime"], axis=1)

      # unix_time_timepoint - 7200 < unix_time_weather < unix_time_timepoint
      df_xwea = self.weather.query('time_in_unix <= @timePoint & time_in_unix >= @timePoint - 7200').tail(1)
      x_wea_temp = df_xwea.drop('DateTime', axis=1)

      x_dep_temp['time_in_unix_SDT'] = x_dep_temp['temp_SDT']
      x_arr_temp['time_in_unix_AAT'] = x_arr_temp['temp_AAT']
      x_wea_temp['time_in_unix'] = x_wea_temp['temp_unix']
      
      x_dep_temp = x_dep_temp.drop('temp_SDT', axis=1)
      x_arr_temp = x_arr_temp.drop('temp_AAT', axis=1)
      x_wea_temp = x_wea_temp.drop('temp_unix', axis=1)
  
      label_x = pd.concat([x_dep_temp.reset_index(drop=True), x_arr_temp.reset_index(drop=True), x_wea_temp.reset_index(drop=True)], axis=1)
      label_x = label_x.fillna(0)
      return label_x, label_y

    def getItem_in_np(self, indexNum):
      x,y = self.getItem_in_df(indexNum)
      return np.array(x).flatten(), np.array(y).flatten()
    
    def getFeatureShape(self):
      return len(self),self.x_col
    
    def getLabelShape(self):
        return len(self),self.y_col