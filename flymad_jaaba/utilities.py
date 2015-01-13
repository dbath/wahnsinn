

import pandas as pd
from pandas import DataFrame
import numpy as np
import glob
import rosbag



def convert_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df
    
def get_absmax(df):
    maximum = df.abs().max(axis=1)  
    return maximum

def bin_data(df, bin_size):
    binned = df.resample(bin_size, how='mean')  
    return binned

def parse_fmftime(namestring):
    fn = namestring.split('/')[-1].split('.fmf')[0]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    fmftime = np.datetime64(fmftime)
    return FLY_ID, fmftime, exp_id

def parse_bagtime(namestring):
    numstr = namestring.split('/')[-1].split('_')[-1].split('.bag')[0].replace('-','')
    bagtime = pd.to_datetime(numstr)
    bagtime = np.datetime64(bagtime)
    return bagtime
    
def match_fmf_and_bag(fmf_filepath, bagdir):
    FLY_ID, fmftime, exp_id = parse_fmftime(fmf_filepath)
    for bag in glob.glob(bagdir + '/*.bag'):
        if abs(parse_bagtime(bag) - fmftime) < np.timedelta64(30000000000, 'ns'):
            matching_bag = bag
            return matching_bag
    if matching_bag == None:
        print "no matching bag file found: ", fmf_filepath
        return None

def match_wide_to_zoom(fmf_filepath, widedir):
    FLY_ID, fmftime, exp_id = parse_fmftime(fmf_filepath)
    for widefmf in glob.glob(widedir + '*wide*.fmf'):
        fly_id, wide_fmf_time, exp_id = parse_fmftime(widefmf)
        if abs(wide_fmf_time - fmftime) < np.timedelta64(30000000000, 'ns'):
            matching_widefmf = widefmf
            return widefmf
        else: matching_widefmf = None
    if matching_widefmf == None:
        print "no matching wide FMF file found: ", fmf_filepath
        return None            
    
def get_positions_from_bag(BAG_FILE):
    bagfile = rosbag.Bag(BAG_FILE)
    baginfo = []
    for topic, msg, t in bagfile.read_messages('/targeter/targeted'):
        baginfo.append((t.secs +t.nsecs*1e-9,msg.fly_x, msg.fly_y))
    baginfo = DataFrame(baginfo, columns=['Timestamp', 'fly_x', 'fly_y'], dtype=np.float64)
    return baginfo
