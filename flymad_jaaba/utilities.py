

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
        
        
