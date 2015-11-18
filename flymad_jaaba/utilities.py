
import os, fnmatch
import pandas as pd
from pandas import DataFrame
import numpy as np
import glob
import rosbag
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math
import subprocess
import sh
import shutil


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def call_command(command, should_I_wait=True):
    foo = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if should_I_wait:
        output = foo.communicate()
    return foo

def delete_temp_files(DIR):
    shutil.rmtree(DIR)
    
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

def parse_calibration_time(namestring):
    numstr = ''.join(namestring.split('/')[-1].split('.')[0].replace('calibration','').split('_'))
    caltime = pd.to_datetime(numstr)
    caltime = np.datetime64(caltime)
    return caltime

    
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
    print "no matching wide FMF file found: ", fmf_filepath, fmftime
    return None            
    
def get_positions_from_bag(BAG_FILE):
    bagfile = rosbag.Bag(BAG_FILE)
    baginfo = []
    for topic, msg, t in bagfile.read_messages('/targeter/targeted'):
        baginfo.append((t.secs +t.nsecs*1e-9,msg.fly_x, msg.fly_y))
    baginfo = DataFrame(baginfo, columns=['Timestamp', 'fly_x', 'fly_y'], dtype=np.float64)
    return baginfo
"""
def binarize_laser_data(BAG_FILE, laser_number):
    bagfile = rosbag.Bag(BAG_FILE)
    laser_current = []
    for topic, msg, t in bagfile.read_messages('/flymad_micro/'+laser_number+'/current'):
        laser_current.append((t.secs +t.nsecs*1e-9,msg.data))
    laser_data = DataFrame(laser_current, columns=['Timestamp', 'Laser_state'], dtype=np.float64)
    laser_data['Laser_state'][laser_data['Laser_state'] > 0.0] = 1.0
    laser_data = convert_timestamps(laser_data)
    return laser_data
"""
def binarize_laser_data(BAG_FILE, laser_number):
    bagfile = rosbag.Bag(BAG_FILE)
    laser_current = []
    for topic, msg, t in bagfile.read_messages('/targeter/targeted'):
        laser_current.append((t.secs +t.nsecs*1e-9,msg.laser_power))
    laser_data = DataFrame(laser_current, columns=['Timestamp', 'Laser_state'], dtype=np.float64)
    laser_data['Laser_state'][laser_data['Laser_state'] > laser_number] = 0.0
    laser_data['Laser_state'][laser_data['Laser_state'] < laser_number] = 0.0
    laser_data['Laser_state'][laser_data['Laser_state'] == laser_number] = 1.0
    laser_data = convert_timestamps(laser_data)
    return laser_data
    
def get_laser_states(BAG_FILE):
    bagfile = rosbag.Bag(BAG_FILE)
    laser_current = []
    config_msg_times = []
    for x in range(3):
        for topic, msg, t in bagfile.read_messages('/flymad_micro/laser' + str(x) + '/configuration'):
            config_msg_times.append((t.secs +t.nsecs*1e-9))
    for topic, msg, t in bagfile.read_messages('/targeter/targeted'):
        laser_current.append((t.secs +t.nsecs*1e-9,msg.laser_power))
    laser_data = DataFrame(laser_current, columns=['Timestamp', 'Laser_state'], dtype=np.float64)
    
    lTime = []
    l0 = []
    l1 = []
    l2 = []
    
    for x in laser_data.index:
        lTime.append(laser_data.Timestamp.ix[x])
        l0.append(int('{0:04b}'.format(int(laser_data['Laser_state'].ix[x]))[-1]))
        l1.append(int('{0:04b}'.format(int(laser_data['Laser_state'].ix[x]))[-2]))
        l2.append(int('{0:04b}'.format(int(laser_data['Laser_state'].ix[x]))[-3]))
    laser_channels = DataFrame({'Timestamp':lTime, 'Laser0_state':l0, 'Laser1_state':l1, 'Laser2_state':l2}, dtype=np.float64)
    laser_channels = laser_channels[laser_channels.Timestamp > min(config_msg_times) ] 
    laser_channels = convert_timestamps(laser_channels)
    return laser_channels


def detect_stim_bouts(datadf, column):
    datadf[column][datadf[column] > 0 ] = 1
        
    maxima = argrelextrema(datadf[column].values, np.greater_equal)[0]
    minima = argrelextrema(datadf[column].values, np.less_equal)[0]
    
    diff = maxima - minima
    ons = argrelextrema(diff, np.greater)[0]
    offs = argrelextrema(diff, np.less)[0]
    number_of_bouts = len(ons)
    if number_of_bouts > 0:
        bout_lengths = []
        for x in range(number_of_bouts):
            bout_lengths.append(1000.0*(datadf.index[offs[x]+1] - datadf.index[ons[x]]).total_seconds())
        bout_duration = int(np.round(np.mean(bout_lengths)))

        first_TS = datadf.index[ons[0]]
        last_TS = datadf.index[offs[-1]]
    else:
        bout_duration = 0
        first_TS = datadf.index[0]
        last_TS = datadf.index[0]
    return number_of_bouts, bout_duration, first_TS, last_TS
        
    
def sendMail(RECIPIENT,SUBJECT,TEXT):
    import sys
    import os
    import re
    from smtplib import SMTP_SSL as SMTP       # this invokes the secure SMTP protocol (port 465, uses SSL)
    # from smtplib import SMTP                  # use this for standard SMTP protocol   (port 25, no encryption)
    from email.MIMEText import MIMEText
    SMTPserver = 'smtp.gmail.com'
    sender =     'danbath@gmail.com'
    destination = [RECIPIENT]

    USERNAME = "danbath"
    PASSWORD = "4Fxahil3"

    # typical values for text_subtype are plain, html, xml
    text_subtype = 'plain'

    
    try:
        msg = MIMEText(TEXT, text_subtype)
        msg['Subject']=       SUBJECT
        msg['From']   = sender # some SMTP servers will do this automatically, not all

        conn = SMTP(SMTPserver)
        conn.set_debuglevel(False)
        conn.login(USERNAME, PASSWORD)
        try:
            conn.sendmail(sender, destination, msg.as_string())
        finally:
            conn.close()
    
    except Exception, exc:
        sys.exit( "mail failed; %s" % str(exc) ) # give a error message
        
def get_calibration_asof_filename(filename):
    if '.bag' in filename:
        ftime = parse_bagtime(filename)
    if '.fmf' in filename:
        _, ftime, __ = parse_fmftime(filename)
    
    for x in sorted(glob.glob('/groups/dickson/home/bathd/wahnsinn/calibrations/*.filtered.yaml')):
        cal_time = parse_calibration_time(x)
        if cal_time <= ftime:
            cal_file = x
    return cal_file
        
        
