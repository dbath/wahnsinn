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
colourlist = ['#000000','#008000','#0032FF','r','c','m','y']

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
    
def sync_jaaba_with_ros(FMF_DIR):

    print "Processing: ", FMF_DIR
    JAABA_CSV = FMF_DIR + '/registered_trx.csv'
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
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
    jaaba_data = bin_data(jaaba_data, binsize)
    jaaba_data['Timestamp'] = jaaba_data.index  #silly pandas bug for subtracting from datetimeindex...
    jaaba_data['Laser_state'][jaaba_data['Laser_state'] > 0] = 1
    jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data['Timestamp'][jaaba_data[jaaba_data['Laser_state'] > 0].index[0]]

    ###    WING EXTENSION    ###
    jaaba_data['maxWingAngle'] = get_absmax(jaaba_data[['Left','Right']])
    
    ### ABDOMINAL BENDING   ###
    jaaba_data[jaaba_data['Length'] > 200] = np.nan  #discard frames with bogus length.  *************
    jaaba_data[jaaba_data['Length'] < 60] = np.nan  #discard frames with bogus length.
    
    
    ###  SAVE DATA ###
    jaaba_data.to_pickle(JAABA + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle')
    
def gather_data(filelist):
    datadf = DataFrame()
    for x in filelist:
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(x)
        fx = pd.read_pickle(x)
        rel = fx[['synced_time','Laser_state', 'maxWingAngle', 'Length', 'Width']]
        rel['group'] = GROUP
        rel['FlyID'] = FLY_ID
        datadf = pd.concat([datadf, rel])
    datadf.to_csv(JAABA + 'rawdata_' + binsize + '.csv', sep=',')
    datadf.to_pickle(JAABA + 'JAR/rawdata_' + binsize + '.pickle')
    #return wingAng 

def group_data(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    grouped = df.groupby(['group', 'synced_time'])
    means = grouped.mean()
    means.to_csv(JAABA + 'mean_' + binsize + '.csv')
    means.to_pickle(JAABA + 'JAR/mean_' + binsize + '.pickle')
    ns = grouped.count()
    ns.to_csv(JAABA + 'n_' + binsize + '.csv')
    sems = grouped.aggregate(lambda x: st.sem(x, axis=None)) 
    sems.to_csv(JAABA + 'sem_' + binsize + '.csv')
    sems.to_pickle(JAABA + 'JAR/sem_' + binsize + '.pickle')   
    return means, sems
"""
def group_body_length(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    grouped = df.groupby(['group', 'synced_time'])
    means = grouped.mean()
    means.to_csv(JAABA + 'body_length_mean_' + binsize + '.csv')
    means.to_pickle(JAABA + 'JAR/body_length_mean_' + binsize + '.pickle')
    sems = grouped.aggregate(lambda x: st.sem(x, axis=None)) 
    sems.to_csv(JAABA + 'body_length_sem_' + binsize + '.csv')
    sems.to_pickle(JAABA + 'JAR/body_length_sem_' + binsize + '.pickle')   
    return means, sems
"""

def plot_data(means, sems, measurement):
    fig = plt.figure()
    group_number = 0
    ax = fig.add_subplot(1,1,1)
    for x in means.index.levels[0]:
        x_values = list((means.ix[x].index)/1e9)
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
    collection = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(means[measurement])), ymax=1.3*(max(means[measurement])), where=means['Laser_state']>0, facecolor='red', alpha=0.3, zorder=10)
    ax.add_collection(collection)
    
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


