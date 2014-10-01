import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pytz import common_timezones

#matlab data should be imported as csv with:
# column 1: seconds since epoch (16 digit precision)
# column 2: left wing angle
# column 3: right wing angle

filename = '/tier2/dickson/bathd/FlyMAD/JAABA_tracking/140927/wing_angles_nano.csv'
binsize = '5s'  # ex: '1s' or '4Min' etc

def convert_timestamps(df):
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df

def get_max(df):
    max_angle = df[['Left','Right']].abs().max(axis=1)  
    return max_angle

def bin_data(df, binsize):
    binned = df.resample('1s', how='mean')  
    return binned
    
wing_data = pd.read_csv(filename, sep=',', names=['Timestamp','Left','Right'])
wing_data = convert_timestamps(wing_data)
wing_data['maxWingAngle'] = get_max(wing_data)
downsampled_wing_data = bin_data(wing_data, binsize)



# extract laser info from bagfile:

import rosbag
bagfile = rosbag.Bag('/groups/dickson/home/bathd/Dropbox/140927_flymad_rosbag_copy/rosbagOut_2014-09-27-14-53-54.bag')
lasertimes = []
for topic, msg, t in bagfile.read_messages('/experiment/laser'):
    lasertimes.append((t.secs +t.nsecs*1e-9,msg.data))
lasertimes = np.array( lasertimes, dtype=[('lasertime',np.float64),('laserstate',np.float32)])

laser_data = DataFrame(lasertimes, columns=['Timestamp', 'Laser_State'])
laser_data = convert_timestamps(laser_data)


wing_data['Laser_state'] = laser_data['Laser_State'].asof(wing_data.index)  #YAY! 
wing_data['Laser_state'] = wing_data['Laser_state'].fillna(value=0)
wing_data['Timestamp'] = wing_data.index  #silly pandas bug for subtracting from datetimeindex...
wing_data['synced_time'] = wing_data['Timestamp'] - wing_data['Timestamp'][wing_data['Laser_state'].idxmax()]
 

