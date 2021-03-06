import time
import os
import glob
import tempfile
import datetime
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
import flymad_jaaba.utilities as utilities
import flymad_jaaba.target_detector as target_detector


    
def fmf2fig(frame, timestamp, colourmap_choice, jaaba, ax):
    image_height, image_width = frame.shape
    LeftAngle = float(jaaba.Left)
    RightAngle = float(jaaba.Right)
    RED_alpha = float(jaaba.Laser2_state)
    IR_alpha = float(jaaba.Laser1_state)
    proximity_val = float((270 - jaaba.dtarget)*((0.8*image_height)/270))#with max dtarget of 270, 2.963 makes max bar height 800.
    
    frame = np.flipud(frame)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.imshow(frame, cmap = colourmap_choice)
    ax.set_xticks([]); ax.set_yticks([])
    np.set_printoptions(precision=2)
    ax.text(0.01*(image_width), 0.01*(image_height), str(np.around(jaaba.ix[0].synced_time / np.timedelta64(1,'s'), 2)) +  's', 
            verticalalignment='top', horizontalalignment='left', 
            color='white', fontsize=22)
    #plot wing angles. use 0.174 multiplier to make 2.3rad equal to 40% of image height.        
    left = plt.Rectangle((image_width-110,image_height/2), 50, LeftAngle*0.174*image_height, color='#FF0000')
    right = plt.Rectangle((image_width-110, image_height/2), 50, RightAngle*0.174*image_height, color='#00FF00')
    RED_ind = plt.Circle((0.1*(image_width), 0.9*(image_height)), 0.05*(image_width), color='r', alpha=RED_alpha)
    IR_ind = plt.Circle((0.25*(image_width), 0.9*(image_height)), 0.05*(image_width), color='k', alpha=IR_alpha)
    #plot proximity
    proxi = plt.Rectangle((image_width-50,((image_height/2) - (proximity_val/2))), 50, proximity_val, color='#0000FF', alpha=1.0)
    
    ax.text(image_width-85, (image_height - 20)/2, 'L', horizontalalignment='center', verticalalignment='bottom',color='white', fontsize=22)
    ax.text(image_width-85, (image_height + 20)/2, 'R', horizontalalignment='center', verticalalignment='top', color='white', fontsize=22)
    #ax.text(0.1*(image_width), 0.85*(image_height), 'Laser:', horizontalalignment='center', verticalalignment='bottom',
    #                                                            color='white', fontsize=16)
    
    ax.add_patch(left)
    ax.add_patch(right)
    ax.add_patch(proxi)
    ax.add_patch(RED_ind)
    ax.add_patch(IR_ind)
    
    return 

def plot_sidebyside(Lfig, Rfig):
    L_height, L_width = Lfig.shape
    R_height, R_width = Rfig.shape
    image_height = max(L_height, R_height)
    image_width = L_width + R_width
    figure = plt.figure(frameon=False)
    figure.set_size_inches(image_height, image_width)
    Lax = plt.Axes(fig, [0., 0., 0.5, 1.])
    Lax.set_axis_off()
    figure.add_axes(Lax)
    Lax.imshow(Lfig, cmap = cm.Greys_r)
    Rax = plt.Axes(fig, [0.5, 0., 1., 1.])
    Rax.set_axis_off()
    figure.add_axes(Rax)
    Rax.imshow(Rfig, cmap = cm.Greys_r)    
    return figure

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
    
def get_absmax(df):
    maximum = df.abs().max(axis=1)  
    return maximum
    
def convert_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df


def sync_jaaba_with_ros(FMF_DIR, BAGS, JAABA):

    print "Processing: ", FMF_DIR
    
    JAABA_CSV               = FMF_DIR + '/registered_trx.csv'
    
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
    
    BAG_FILE                = utilities.match_fmf_and_bag(FMF_DIR, BAGS)
    
    WIDE_FMF                = utilities.match_wide_to_zoom(FMF_DIR, JAABA)
    
    jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
    jaaba_data = utilities.convert_timestamps(jaaba_data)
    
    # ALIGN LASER STATE DATA
    jaaba_data['Laser1_state'] = binarize_laser_data(BAG_FILE, 'laser1')['Laser_state'].asof(jaaba_data.index).fillna(value=0)  #YAY! 
    jaaba_data['Laser2_state'] = binarize_laser_data(BAG_FILE, 'laser2')['Laser_state'].asof(jaaba_data.index).fillna(value=0)
    
    # COMPUTE AND ALIGN DISTANCE TO NEAREST TARGET
    targets = target_detector.TargetDetector(WIDE_FMF, FMF_DIR)
    targets.plot_targets_on_background()
    jaaba_data['dtarget'] = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget'].asof(jaaba_data.index).fillna(value=0)
    
    jaaba_data['Timestamp'] = jaaba_data.index  #silly pandas bug for subtracting from datetimeindex...
    jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data['Timestamp'][jaaba_data[jaaba_data['Laser2_state'] > 0].index[0]]    
    
    #DISCARD BOGUS WING ANGLE VALUES:
    jaaba_data['Left'][jaaba_data['Left'] < -2.3 ]= np.nan   #2.09 for 120degrees
    jaaba_data['Right'][jaaba_data['Right'] > 2.3] = np.nan
    
    jaaba_data['maxWingAngle'] = get_absmax(jaaba_data[['Left','Right']])
    

    return jaaba_data

def plot_window_of_data(timestamp, data, windowsize, ax, measurement):
 
    timestamp = pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
    window_td = datetime.timedelta(0,windowsize,0)
    first_ts = (timestamp - window_td)
    last_ts = (timestamp + window_td) 
    window_data = data[(data.index >= first_ts) & (data.index <= last_ts)]
    #print (window_data['Timestamp'] - timestamp).values, '\t', window_data[measurement].values
    x_values = (window_data['Timestamp'] - timestamp).values / np.timedelta64(1,'s')
    y_values = window_data[measurement].values
    print x_values[100], '\t', y_values[100]
    """
    for w in window_data['Timestamp']:
        #print (w-pd.to_datetime(0).tz_localize('UTC').tz_convert('US/Eastern')).total_seconds()
        x_values.append((w-pd.to_datetime(0).tz_localize('UTC').tz_convert('US/Eastern')).total_seconds())
        y_values.append(window_data[measurement][window_data[window_data['Timestamp'] == w]].values[0])
        #print window_data[measurement][window_data[window_data['Timestamp'] == w]].values[0]
    """
    plt.plot(x_values, y_values, color='b', linewidth=2)
    plt.axvline((0), color='#00FF00', linewidth=2)
    #call before this function: AXNAME = fig.add_subplot(x,x,x) \n fig.add_axes(AXNAME)
    
    #ax.set_xlim(first_ts, last_ts)
    ax.set_ylim(0.85*(min(data[measurement])),1.15*(max(data[measurement])))
    ax.set_ylabel(measurement , fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=16)
    
    return 
    
    
def get_frame_number_at_or_before_timestamp(fmf_object, timestamp): #stolen and modified from motmot!
    tss = fmf_object.get_all_timestamps()
    at_or_before_timestamp_cond = tss <= timestamp
    nz = np.nonzero(at_or_before_timestamp_cond)
    if len(nz)==0:
        raise ValueError("no frames at or before timestamp given")
    fno = nz[0][-1]
    return fno

def get_data(fmf_file):
    for x in glob.glob(fmf_file + '/*.fmf'):
        fmf =  FMF.FlyMovie(x)
        
    if not os.path.exists(savedir + 'stashed_data.pickle'):
        data = sync_jaaba_with_ros(fmf_file, BAGS, WIDE_DIR)
        data.to_pickle(savedir + 'stashed_data.pickle')
    data = pd.read_pickle(savedir + 'stashed_data.pickle')
    zero_timestamp = data['Timestamp'][data[data['Laser2_state'] > 0].index[0]]
    zero_ts_float = (np.datetime64(zero_timestamp.to_datetime()) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    frame_at_t0 = get_frame_number_at_or_before_timestamp(fmf, zero_ts_float)
    print frame_at_t0
    return fmf, data, frame_at_t0
    
    
def make_png_set(fmf_dir, save_dir):

    
    if not os.path.exists(save_dir + '/temp_png'):
        os.makedirs(save_dir + '/temp_png')

    fmf, jaaba_data, t0 = get_data(fmf_dir)
    print jaaba_data.index[t0]
    image_height, image_width = fmf.get_frame(0)[0].shape

    for frame_number in range(3500):
                
        if os.path.exists(save_dir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        
        fig = plt.figure(figsize=(image_width/100, image_height*1.1/100), dpi=200.399 )
        
        #ax1 = fig.add_subplot(2,1,1)
        ax1 = plt.subplot2grid((11,10), (0,0), colspan=10, rowspan=10)
       
        fig.add_axes(ax1)
        
        frame,timestamp = fmf.get_frame((frame_number + t0 - 225))     
        print pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
        jaaba_datum = jaaba_data[jaaba_data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
        fmf2fig(frame, timestamp, cm.Greys_r, jaaba_datum, ax1)
        
        #ax2 = fig.add_subplot(2,1,2)
        ax2 = plt.subplot2grid((11,10), (10,0), colspan=10)
        fig.add_axes(ax2)
        plot_window_of_data(timestamp, jaaba_data, 20, ax2, 'dtarget')

        
        
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')
    
    sendMail('bathd@janelia.hhmi.org','movie is finished', ('Your side by side video has finished.'))


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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fmf', type=str, required=True,
                        help='path to fmf') 
    parser.add_argument('--bagdir', type=str, required=True,
                        help='path to bag files') 
    parser.add_argument('--widedir', type=str, required=True,
                        help='path to directory containing wide fmfs') 
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save directory') 
    args = parser.parse_args()

    fmf = args.fmf
    savedir = args.savedir    
    BAGS = args.bagdir
    WIDE_DIR = args.widedir
    
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

    baglist = []
    for bag in glob.glob(BAGS + '/*.bag'):
        bagtimestamp = parse_bagtime(bag)
        baglist.append((bag, bagtimestamp))
    bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
    bagframe.index = pd.to_datetime(bagframe['Timestamp'])
    bagframe = bagframe.sort()
    bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')


    make_png_set(fmf, savedir)

    """    
    ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4


    ffmpeg = script

    -f = 

    image2 = 

    -r 1/5 =  framerate of 5s per image. for 15FPS, use -r 15
    """

