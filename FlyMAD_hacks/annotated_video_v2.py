import time
import os
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
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--videodir', type=str, required=True,
                        help='directory of fmf and JAABA csv files')  
parser.add_argument('--startframe', type=int, required=False, default=0,
                        help='frame from fmf to start')
parser.add_argument('--endframe', type=int, required=False, default=None,
                        help='frame from fmf to end')
#parser.add_argument('--binsize', type=str, required=True,
#                      help='integer and unit, such as "5s" or "4Min"')

args = parser.parse_args()

VIDEO_DIR = args.videodir
BAGS = VIDEO_DIR + 'BAGS/'#VIDEO_DIR.rsplit('/',2)[0] + '/BAGS/'
FIRST = args.startframe
LAST = args.endframe

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
    
def binarize_laser_data(BAG_FILE, laser_number):
    bagfile = rosbag.Bag(BAG_FILE)
    laser_current = []
    for topic, msg, t in bagfile.read_messages('/flymad_micro/'+laser_number+'/current'):
        laser_current.append((t.secs +t.nsecs*1e-9,msg.data))
    laser_data = DataFrame(laser_current, columns=['Timestamp', 'Laser_state'], dtype=np.float64)
    laser_data['Laser_state'][laser_data['Laser_state'] > 0] = 1.0
    laser_data = convert_timestamps(laser_data)
    return laser_data
    
def sync_jaaba_with_ros(JAABA_path, FMF_path):
    JAABA_CSV = JAABA_path
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_path)
    BAG_FILE = match_fmf_and_bag(FMF_TIME)
    
    jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
    #jaaba_data[abs(jaaba_data['Left']) > 1.57] = 0.0
    #jaaba_data[abs(jaaba_data['Right']) > 1.57] = 0.0
    jaaba_data = convert_timestamps(jaaba_data)
    
    jaaba_data['Laser1_state'] = binarize_laser_data(BAG_FILE, 'laser1')['Laser_state'].asof(jaaba_data.index).fillna(value=0)  #YAY! 
    jaaba_data['Laser2_state'] = binarize_laser_data(BAG_FILE, 'laser2')['Laser_state'].asof(jaaba_data.index).fillna(value=0)
    
    jaaba_data['Timestamp'] = jaaba_data.index  #silly pandas bug for subtracting from datetimeindex...
    jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data['Timestamp'][jaaba_data[jaaba_data['Laser2_state'] > 0].index[0]]
    
    return jaaba_data 
    
def fmf2fig(frame, timestamp, colourmap_choice, jaaba):
    image_width, image_height = frame.shape
    #elapsed_time = int(timestamp - starting_timestamp)
    LeftAngle = float(jaaba.Left)
    RightAngle = float(jaaba.Right)
    RED_alpha = float(jaaba.Laser2_state)
    IR_alpha = float(jaaba.Laser1_state)
    current_time =  jaaba.synced_time.asof(pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern'))
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
            color='white', fontsize=16)
    left = plt.Rectangle((image_width-60,image_height/2), 50, LeftAngle*300, color='#FF0000')
    right = plt.Rectangle((image_width-60, image_height/2), 50, RightAngle*300, color='#00FF00')
    RED_ind = plt.Circle((0.1*(image_width), 0.9*(image_height)), 0.05*(image_width), color='r', alpha=RED_alpha)
    IR_ind = plt.Circle((0.25*(image_width), 0.9*(image_height)), 0.05*(image_width), color='k', alpha=IR_alpha)
    ax.text(image_width-30, (image_height - 20)/2, 'L', horizontalalignment='center', verticalalignment='bottom',
                                                                color='white', fontsize=22)
    ax.text(image_width-30, (image_height + 20)/2, 'R', horizontalalignment='center', verticalalignment='top',
                                                                color='white', fontsize=22)
    #ax.text(0.1*(image_width), 0.85*(image_height), 'Laser:', horizontalalignment='center', verticalalignment='bottom',
    #                                                            color='white', fontsize=16)
    
    ax.add_patch(left)
    ax.add_patch(right)
    ax.add_patch(RED_ind)
    ax.add_patch(IR_ind)
    
    return figure




baglist = []
for bag in glob.glob(BAGS + '/*.bag'):
    bagtimestamp = parse_bagtime(bag)
    baglist.append((bag, bagtimestamp))
bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
bagframe.index = pd.to_datetime(bagframe['Timestamp'])
bagframe = bagframe.sort()
bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')


def make_png_set(_VIDEO_DIR):
    if os.path.exists(_VIDEO_DIR + '/flymad_annotated.mp4'):  ###should use complete mp4 file here
        print 'skipping already finished file:', _VIDEO_DIR
        return
    
    
    if not os.path.exists(_VIDEO_DIR + '/temp_png'):
        os.makedirs(_VIDEO_DIR + '/temp_png')

    for x in glob.glob(_VIDEO_DIR + '/*.fmf'):
        fmf = FMF.FlyMovie(x)
        jaaba_data = sync_jaaba_with_ros((_VIDEO_DIR + '/registered_trx.csv'), x)
    

    if LAST == None:
        _LAST = fmf.get_n_frames()
    else:
        _LAST = LAST

    for frame_number in range((fmf.get_n_frames() - FIRST - (fmf.get_n_frames() - _LAST))):

        if os.path.exists(_VIDEO_DIR + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        frame, timestamp = fmf.get_frame(frame_number + FIRST)
        #print frame_number #pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern'), '\t', frame_number
        
        jaaba_datum = jaaba_data[jaaba_data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
 
        fmf2fig(frame, timestamp, cm.Greys_r, jaaba_datum)
        #plt.show()
        plt.savefig(_VIDEO_DIR + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')
    
    sendMail('bathd@janelia.hhmi.org','movie is finished', (x + ' has finished processing.'))
    
for x in glob.glob(VIDEO_DIR + '*costim*' + '*zoom*'):
    print "processing:", x
    make_png_set(x)
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

"""    
ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4


ffmpeg = script

-f = 

image2 = 

-r 1/5 =  framerate of 5s per image. for 15FPS, use -r 15
"""

