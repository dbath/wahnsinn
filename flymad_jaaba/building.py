import time
import os.path
import glob
import tempfile
import shutil
import re
import collections
import operator
import multiprocessing
import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import pandas as pd
from pandas import DataFrame
import numpy as np
import rosbag
import rosbag_pandas
import benu.benu
import benu.utils

from PIL import Image, ImageStat

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
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    TIME = TIME.split('.')[0]
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def parse_bagtime(namestring):
    numstr = namestring.split('/')[-1].split('_')[-1].split('.bag')[0].replace('-','')
    bagtime = pd.to_datetime(numstr)
    return bagtime
    
def match_fmf_and_bag(fmftime):
    fmftime64 = np.datetime64(fmftime)
    bagtime = bagframe['Timestamp'].asof(fmftime)
    if fmftime64 - bagtime > np.timedelta64(30000000000, 'ns'):
        print "ERROR: fmf is more than 30 seconds younger than bagfile: ", fmftime
    bagfile = bagframe['Filepath'].asof(fmftime)
    return bagfile

def sync_jaaba_with_ros(JAABA_path, FMF_path):
    JAABA_CSV = JAABA_path
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_path)
    BAG_FILE = match_fmf_and_bag(FMF_TIME)
    
    jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
    jaaba_data = convert_timestamps(jaaba_data)

    # extract laser info from bagfile:

    bagfile = rosbag.Bag(BAG_FILE)
    lasertimes = []
    for topic, msg, t in bagfile.read_messages('/flymad_micro/position'):
        lasertimes.append((t.secs +t.nsecs*1e-9,msg.laser))
    laser_data = DataFrame(lasertimes, columns=['Timestamp', 'Laser_State'], dtype=np.float64)
    laser_data = convert_timestamps(laser_data)
    
    jaaba_data['Laser_state'] = laser_data['Laser_State'].asof(jaaba_data.index)  #YAY! 
    jaaba_data['Laser_state'] = jaaba_data['Laser_state'].fillna(value=0)
    #jaaba_data = bin_data(jaaba_data, binsize)
    jaaba_data['Timestamp'] = jaaba_data.index  #silly pandas bug for subtracting from datetimeindex...
    jaaba_data['Laser_state'][jaaba_data['Laser_state'] > 0] = 1
    jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data['Timestamp'][jaaba_data[jaaba_data['Laser_state'] > 0].index[0]]
    
    """
    ###    WING EXTENSION    ###
    jaaba_data['maxWingAngle'] = get_absmax(jaaba_data[['Left','Right']])
    
    ### ABDOMINAL BENDING   ###
    jaaba_data[jaaba_data['Length'] > 200] = np.nan  #discard frames with bogus length.  *************
    jaaba_data[jaaba_data['Length'] < 60] = np.nan  #discard frames with bogus length.
    return jaaba_data
    """
    return jaaba_data 
    
def fmf2fig(frame, timestamp, colourmap_choice, jaaba):
    #elapsed_time = int(timestamp - starting_timestamp)
    LeftAngle = float(jaaba.Left)
    RightAngle = float(jaaba.Right)
    laser_alpha = float(jaaba.Laser_state)
    current_time =  jaaba.synced_time.asof(pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern'))
    if current_time >= np.timedelta64(180, 's'):
        laser_colour='k'
    else:
        laser_colour='r'
    frame = np.flipud(frame)
    figure = plt.figure(figsize=(image_width/100, image_height/100), dpi=200.399 )
    ax = figure.add_subplot(1,1,1)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.imshow(frame, cmap = colourmap_choice)
    ax.set_xticks([]); ax.set_yticks([])
    np.set_printoptions(precision=2)
    ax.text(0.01*(image_width), 0.01*(image_height), str(np.around(jaaba.ix[0].synced_time / np.timedelta64(1,'s'), 2)) +  's', 
            verticalalignment='top', horizontalalignment='left', 
            color='white', fontsize=10) #pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
    left = plt.Rectangle((image_width-60,image_height/2), 50, LeftAngle*300, color='#FF0000')
    right = plt.Rectangle((image_width-60, image_height/2), 50, RightAngle*300, color='#00FF00')
    laser_ind = plt.Circle((0.1*(image_width), 0.9*(image_height)), 0.05*(image_width), color=laser_colour, alpha=laser_alpha)
    ax.text(image_width-30, (image_height - 20)/2, 'L', horizontalalignment='center', verticalalignment='bottom',
                                                                color='white', fontsize=22)
    ax.text(image_width-30, (image_height + 20)/2, 'R', horizontalalignment='center', verticalalignment='top',
                                                                color='white', fontsize=22)
    #ax.text(0.1*(image_width), 0.85*(image_height), 'Laser:', horizontalalignment='center', verticalalignment='bottom',
    #                                                            color='white', fontsize=16)
    
    ax.add_patch(left)
    ax.add_patch(right)
    ax.add_patch(laser_ind)
    man
    return figure

BAGS = '/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags'
baglist = []
for bag in glob.glob(BAGS + '/*.bag'):
    bagtimestamp = parse_bagtime(bag)
    baglist.append((bag, bagtimestamp))
bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
bagframe.index = pd.to_datetime(bagframe['Timestamp'])
bagframe = bagframe.sort()
bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')



trx = pd.read_csv('/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags/registered_trx.csv',sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    
fname = '/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags/DB199-UC-RIR_zoom_20141005_150045.fmf' 
fileName = '/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags/test.png'
fmf = FMF.FlyMovie(fname)

jaaba_data = sync_jaaba_with_ros('/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags/registered_trx.csv', '/tier2/dickson/bathd/FlyMAD/sample_fmfs_and_bags/DB199-UC-RIR_zoom_20141005_150045.fmf')

_frame, _timestamp = fmf.get_frame(0)
image_width, image_height = _frame.shape

wide_fmf = FMF.FlyMovie('_PATH TO WIDE FMF_______')

fmf_dict = {}
for frame_number in range(wide_fmf.get_n_frames()):
    frame, timestamp = fmf.get_frame(frame_number)
    fmf_dict[timestamp] = DataFrame(frame)
pn = Panel(fmf_dict)
maxproj = pn.max(axis=0)
mask = maxproj / 255
mask[mask >0.5] = 1
mask[mask <= 0.5] = 0
eroded = ndimage.binary_erosion(mask, iterations=4).astype(np.float32)


for frame_number in range(1271,7200):#fmf.get_n_frames()):
    if os.path.exists(BAGS + '/temp_png/_tmp%05d.png'%(frame_number-1271)):
        continue
    frame, timestamp = fmf.get_frame(frame_number)
    print frame_number #pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern'), '\t', frame_number
    
    jaaba_datum = jaaba_data[jaaba_data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
 
    fmf2fig(frame, timestamp, cm.Greys_r, jaaba_datum)
    #plt.show()
    plt.savefig(BAGS + '/temp_png/_tmp%05d.png'%(frame_number-1271), bbox_inches='tight', pad_inches=0)
    plt.close('all')
    

    



