import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from pytz import common_timezones
import rosbag
import argparse
import glob
import os
from scipy import stats as st

#matlab data should be imported as csv with:
# column 1: seconds since epoch (16 digit precision)
# column 2: left wing angle
# column 3: right wing angle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--jaabadir', type=str, required=True,
                        help='directory of JAABA csv files')  
#parser.add_argument('--outputdir', type=str, required=True,
#                        help='directory to store analysis')
#parser.add_argument('--bagdir', type=str, required=True,
#                        help='directory of bag files')
parser.add_argument('--binsize', type=str, required=True,
                        help='integer and unit, such as "5s" or "4Min"')

args = parser.parse_args()

JAABA = args.jaabadir
BAGS = JAABA + 'BAGS'

#OUTPUT = args.outputdir

binsize = (args.binsize)
print "BINSIZE: ", binsize
colourlist = ['#008000','#0032FF','r','c','m','y', '#000000']

#filename = '/tier2/dickson/bathd/FlyMAD/JAABA_tracking/140927/wing_angles_nano.csv'
#binsize = '5s'  # ex: '1s' or '4Min' etc
#BAG_FILE = '/groups/dickson/home/bathd/Dropbox/140927_flymad_rosbag_copy/rosbagOut_2014-09-27-14-53-54.bag'


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
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def parse_bagtime(namestring):
    numstr = namestring.split('/')[-1].split('_')[-1].split('.bag')[0].replace('-','')
    bagtime = pd.to_datetime(numstr)
    return bagtime
    
def match_fmf_and_bag(fmftime):
    #print 'from mat_fmf_and_bag: ', fmftime
    fmftime64 = np.datetime64(fmftime)
    bagtime = bagframe['Timestamp'].asof(fmftime)
    #print "bagtime: ", bagtime
    if fmftime64 - bagtime > np.timedelta64(30000000000, 'ns'):
        print "ERROR: fmf is more than 30 seconds younger than bagfile: ", fmftime
    bagfile = bagframe['Filepath'].asof(fmftime)
    return bagfile
    
    
def raster(event_times_list, color='k'):
    """
    Creates a raster plot
     
    Parameters
    ----------
    event_times_list : iterable
    a list of event time iterables
    color : string
    color of vlines
     
    Returns
    -------
    ax : an axis containing the raster plot
    """#def
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
        plt.ylim(.5, len(event_times_list) + .5)
    return ax    

def binarize_laser_data(BAG_FILE, laser_number):
    bagfile = rosbag.Bag(BAG_FILE)
    laser_current = []
    for topic, msg, t in bagfile.read_messages('/flymad_micro/'+laser_number+'/current'):
        laser_current.append((t.secs +t.nsecs*1e-9,msg.data))
    laser_data = DataFrame(laser_current, columns=['Timestamp', 'Laser_state'], dtype=np.float64)
    laser_data['Laser_state'][laser_data['Laser_state'] > 0] = 1.0
    laser_data = convert_timestamps(laser_data)
    return laser_data
    
def sync_jaaba_with_ros(FMF_DIR):

    print "Processing: ", FMF_DIR
    JAABA_CSV = FMF_DIR + '/registered_trx.csv'
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
    BAG_FILE = match_fmf_and_bag(FMF_TIME)
    
    jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
    jaaba_data = convert_timestamps(jaaba_data)

    
    jaaba_data['Laser1_state'] = binarize_laser_data(BAG_FILE, 'laser1')['Laser_state'].asof(jaaba_data.index).fillna(value=0)  #YAY! 
    jaaba_data['Laser2_state'] = binarize_laser_data(BAG_FILE, 'laser2')['Laser_state'].asof(jaaba_data.index).fillna(value=0)
    
    jaaba_data['Timestamp'] = jaaba_data.index #silly pandas bug for subtracting from datetimeindex...
    
    jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data.Timestamp[jaaba_data.Laser1_state > 0].index[0]
    jaaba_data.index = jaaba_data.synced_time
    jaaba_data.index = pd.to_datetime(jaaba_data.index)

    ###    WING EXTENSION    ###
    jaaba_data['maxWingAngle'] = get_absmax(jaaba_data[['Left','Right']])
    #jaaba_data[jaaba_data['maxWingAngle'] > 1.57] = np.nan
    
    ### ABDOMINAL BENDING   ###
    jaaba_data[jaaba_data['Length'] > 200] = np.nan  #discard frames with bogus length.  *************
    jaaba_data[jaaba_data['Length'] < 60] = np.nan  #discard frames with bogus length.
    
    
    jaaba_data = bin_data(jaaba_data, binsize)
    ###  SAVE DATA ###
    jaaba_data.to_pickle(JAABA + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle')
    
def gather_data(filelist):
    datadf = DataFrame()
    for x in filelist:
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(x)
        fx = pd.read_pickle(x)
        rel = fx[['Laser1_state', 'Laser2_state', 'maxWingAngle', 'Length', 'Width']]
        rel['group'] = GROUP
        rel['FlyID'] = FLY_ID
        datadf = pd.concat([datadf, rel])
    datadf.to_csv(JAABA + 'rawdata_' + binsize + '.csv', sep=',')
    datadf.to_pickle(JAABA + 'JAR/rawdata_' + binsize + '.pickle')
    #return wingAng 

def group_data(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    grouped = df.groupby(['group', df.index])
    means = grouped.mean()
    means.to_csv(JAABA + 'mean_' + binsize + '.csv')
    means.to_pickle(JAABA + 'JAR/mean_' + binsize + '.pickle')
    ns = grouped.count()
    ns.to_csv(JAABA + 'n_' + binsize + '.csv')
    sems = grouped.aggregate(lambda x: st.sem(x, axis=None)) 
    sems.to_csv(JAABA + 'sem_' + binsize + '.csv')
    sems.to_pickle(JAABA + 'JAR/sem_' + binsize + '.pickle')   
    return means, sems

def plot_data(means, sems, measurement):
    fig = plt.figure()
    group_number = 0
    ax = fig.add_subplot(1,1,1)
    for x in means.index.levels[0]:
        x_values = []
        for w in means.ix[x].index:
            x_values.append((w-pd.to_datetime(0)).total_seconds())
        #x_values = list((means.ix[x].index - pd.to_datetime(0)).total_seconds())
        y_values = list(means.ix[x][measurement])
        psems = list(sems.ix[x][measurement])
        nsems = list(-1*(sems.ix[x][measurement]))
        top_errbar = tuple(map(sum, zip(psems, y_values)))
        bottom_errbar = tuple(map(sum, zip(nsems, y_values)))
        p = plt.plot(x_values, y_values, linewidth=8, zorder=100,
                        linestyle = '-',
                        color=colourlist[group_number],
                        label=x) 
        q = plt.fill_between(x_values, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.15, 
                            zorder=90,
                            color=colourlist[group_number],
                            )
        group_number += 1
    ax.set_xlim((np.amin(x_values),np.amax(x_values)))
    ax.set_ylim(0.85*(min(means[measurement])),1.15*(max(y_values)))
    if 'maxWingAngle' in measurement:
        ax.set_ylabel('Mean maximum wing angle (rad)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
    else:
        ax.set_ylabel('Mean ' + measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Time (s)', fontsize=16)
    laser_1 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(means[measurement])), ymax=1.3*(max(means[measurement])), where=means['Laser1_state'] > 0, facecolor='#999999', alpha=1.0, zorder=10) #green b2ffb2
    ax.add_collection(laser_1)
    laser_2 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(means[measurement])), ymax=1.3*(max(means[measurement])), where=means['Laser2_state'] > 0, facecolor='#FFB2B2', alpha=1.0, zorder=10) #red FFB2B2
    ax.add_collection(laser_2)
    laser_1_2 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(means[measurement])), ymax=1.3*(max(means[measurement])), where=((means['Laser1_state'] > 0) & (means['Laser2_state'] > 0)) , facecolor='#FF9999', alpha=1.0, zorder=11) #yellow FFFFB2
    ax.add_collection(laser_1_2)
    
    ax.legend()
    plt.show()
    if 'maxWingAngle' in measurement:
        fig.savefig(JAABA + 'mean_max_wing_angle_' + binsize + '_bins.svg', bbox_inches='tight')
    else:
        fig.savefig(JAABA + 'mean_' + measurement + '_' + binsize + '_bins.svg', bbox_inches='tight')
def plot_rasters(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    grouped = df.groupby(['group','synced_time'])
    fig = plt.figure()
    group_number = 1
    for x in set(grouped.mean().index.levels[0]):
        ax = fig.add_subplot(1,len(set(grouped.mean().index.levels[0])), group_number)
        group_number +=1
        actions = []
        


baglist = []
for bag in glob.glob(BAGS + '/*.bag'):
    bagtimestamp = parse_bagtime(bag)
    baglist.append((bag, bagtimestamp))
bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
bagframe.index = pd.to_datetime(bagframe['Timestamp'])
bagframe = bagframe.sort()
bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')

if not os.path.exists(JAABA + 'JAR') ==True:
    print "MAKING A JAR"
    os.makedirs(JAABA + 'JAR')
####assess wing data processing:
processed_filelist = glob.glob(JAABA+ 'JAR/*' + binsize + '_fly.pickle')
print len(processed_filelist), " files found."
if os.path.isfile(JAABA + 'JAR/mean_' + binsize + '.pickle') == True:
    print "Using pickled grouped data."
    means =  pd.read_pickle(JAABA + 'JAR/mean_' + binsize + '.pickle')
    sems = pd.read_pickle(JAABA + 'JAR/sem_' + binsize + '.pickle')
    
elif os.path.isfile(JAABA+'JAR/wing_angles_raw_' + binsize + '.pickle') == True:
    print "Using pickled rawfile."
    means, sems = group_data(JAABA + 'JAR/rawdata_' + binsize + '.pickle')
elif (len(processed_filelist) <= 1):
    print "Processing data from scratch"
    for directory in glob.glob(JAABA + '*zoom*'):
        sync_jaaba_with_ros(directory)
        
    gather_data(glob.glob(JAABA + 'JAR/*' + binsize + '_fly.pickle'))
    means, sems = group_data(JAABA + 'JAR/rawdata_' + binsize + '.pickle')
else:
    print "Using processed fly data"
    gather_data(glob.glob(JAABA + 'JAR/*' + binsize + '_fly.pickle'))
    means, sems = group_data(JAABA + 'JAR/rawdata_' + binsize + '.pickle')
    
plot_data(means, sems, 'maxWingAngle')    
plot_data(means, sems, 'Length')
plot_data(means, sems, 'Width')


