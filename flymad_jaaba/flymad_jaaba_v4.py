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
import flymad_jaaba.utilities as utilities
import flymad_jaaba.target_detector as target_detector

#matlab data should be imported as csv with:
# column 1: seconds since epoch (16 digit precision)
# column 2: left wing angle
# column 3: right wing angle


NANOSECONDS_PER_SECOND = 1000000000

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
    laser_data['Laser_state'][laser_data['Laser_state'] > 0.0] = 1.0
    laser_data = convert_timestamps(laser_data)
    return laser_data
    
def sync_jaaba_with_ros(FMF_DIR):

    print "Processing: ", FMF_DIR
    
    JAABA_CSV               = FMF_DIR + '/registered_trx.csv'
    
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
    
    BAG_FILE                = match_fmf_and_bag(FMF_TIME)
    
    WIDE_FMF                = utilities.match_wide_to_zoom(FMF_DIR, JAABA)
    
    jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
    jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
    jaaba_data = convert_timestamps(jaaba_data)

    # ALIGN LASER STATE DATA
    jaaba_data['Laser1_state'] = binarize_laser_data(BAG_FILE, 'laser1')['Laser_state'].asof(jaaba_data.index).fillna(value=0)  #YAY! 
    jaaba_data['Laser2_state'] = binarize_laser_data(BAG_FILE, 'laser2')['Laser_state'].asof(jaaba_data.index).fillna(value=0)
    
    # COMPUTE AND ALIGN DISTANCE TO NEAREST TARGET
    targets = target_detector.TargetDetector(WIDE_FMF, FMF_DIR)
    targets.plot_targets_on_background()
    targets.plot_trajectory_on_background(BAG_FILE)
    jaaba_data['dtarget'] = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget'].asof(jaaba_data.index).fillna(value=0)
    
    
    jaaba_data['Timestamp'] = jaaba_data.index #silly pandas bug for subtracting from datetimeindex...
    try:
        jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data.Timestamp[(jaaba_data.Laser2_state + jaaba_data.Laser1_state) > 0.001].index[0]
    except:
        print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
        jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data.Timestamp.index[0]
    jaaba_data.index = jaaba_data.synced_time
    jaaba_data.index = pd.to_datetime(jaaba_data.index)

    ###    WING EXTENSION    ###
    jaaba_data['maxWingAngle'] = get_absmax(jaaba_data[['Left','Right']])
    jaaba_data[jaaba_data['maxWingAngle'] > 2.1] = np.nan
    
    ### ABDOMINAL BENDING   ###
    jaaba_data[jaaba_data['Length'] > 110] = np.nan  #discard frames with bogus length.  *************
    jaaba_data[jaaba_data['Length'] < 60] = np.nan  #discard frames with bogus length.
    
    trace = plot_single_trace(jaaba_data)
    trace.savefig(JAABA + 'TRACES/' + FLY_ID + '.png')
    plt.close('all')
    
    ###FIRST COURTSHIP BOUT AFTER STIMULUS###
    
    #courted, latency = latency_measures(jaaba_data)
    
    
    if 'binsize' in globals():
        jaaba_data = bin_data(jaaba_data, binsize)
        jaaba_data.to_pickle(JAABA + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle')
    else:
        return jaaba_data, courted, latency
    
def latency_measures(jaaba_data): #takes input from unbinned jaaba_data (during sync...)
      poststim = jaaba_data[(jaaba_data.index > jaaba_data[(jaaba_data.Laser2_state + jaaba_data.Laser1_state) > 0.001].index[-1])]
      courting = poststim[(poststim.maxWingAngle >= 0.523598776)  & (poststim.dtarget <= 50 )]
      try: 
        latency = (courting.index[0] - poststim.index[0]).total_seconds()
        courted = 1
      except:
        courted = 0
        latency = (poststim.index[-1] - poststim.index[0]).total_seconds()
      return courted, latency
      
      
def plot_latency_to_courtship(list_of_latencies):
    df = DataFrame(list_of_latencies, columns=['genotype', 'latency'])
    means = df.groupby('genotype').mean()
    fig = plt.figure()
    
    
def gather_data(filelist):
    datadf = DataFrame()
    for x in filelist:
        FLY_ID = x.split('/')[-1].split('_fly.')[0]
        EXP_ID, DATE, TIME = FLY_ID.split('_', 4)[0:3]
        fx = pd.read_pickle(x)
        fx = fx[fx.columns]
        PC_wing = fx[(fx.index >= pd.to_datetime(THRESH_ON*NANOSECONDS_PER_SECOND)) & (fx.index <= pd.to_datetime(THRESH_OFF*NANOSECONDS_PER_SECOND))]['maxWingAngle']
        WEI = float(PC_wing[PC_wing >= 0.524].count()) / float(PC_wing.count())
        if WEI < WEI_THRESHOLD:
            print FLY_ID, " excluded from analysis, with wing extension index: " , WEI , "."
            continue
        fx['group'] = EXP_ID
        fx['FlyID'] = FLY_ID
        datadf = pd.concat([datadf, fx])
    datadf.to_csv(JAABA + HANDLE + '_rawdata_' + binsize + '.csv', sep=',')
    datadf.to_pickle(JAABA + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    #return wingAng return FLY_ID, fmftime, exp_id exp_id, CAMN, DATE, TIME = fn.split('_', 3)

def pool_genotypes(df):
    for item in POOL_CONTROL:
        df = df.replace(to_replace=item, value='pooled controls')
    for item in POOL_EXP:
        df = df.replace(to_replace=item, value='experiment')       
    return df
    
def group_data(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    df = pool_genotypes(df)
    grouped = df.groupby(['group', df.index])
    means = grouped.mean()
    ns = grouped.count()
    sems = grouped.aggregate(lambda x: st.sem(x, axis=None))
    means.to_csv(JAABA + HANDLE + '_mean_' + binsize + '.csv')
    means.to_pickle(JAABA + 'JAR/'+HANDLE+ '_mean_' + binsize + '.pickle')
    ns.to_csv(JAABA + HANDLE + '_n_' + binsize + '.csv') 
    ns.to_pickle(JAABA + 'JAR/' + HANDLE + '_n_' + binsize + '.pickle')   
    sems.to_csv(JAABA + HANDLE + '_sem_' + binsize + '.csv')
    sems.to_pickle(JAABA + 'JAR/' + HANDLE + '_sem_' + binsize + '.pickle')   
    
    return means, sems, ns

def plot_single_trace(jaaba_data):
    fig = plt.figure()
    measurements = ['maxWingAngle', 'Length', 'Width', 'dtarget']
    x_values=[]
    for w in jaaba_data.index:
        x_values.append((w-pd.to_datetime(0)).total_seconds())
    
    for m in range(len(measurements)):
        ax = fig.add_subplot(len(measurements), 1,(m+1))
        y_values = jaaba_data[measurements[m]]
        p = plt.plot(x_values, y_values, linewidth=2, zorder=100)
        laser_1 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(y_values.min()), ymax=1.3*(y_values.max()), where=jaaba_data['Laser1_state'] > 0, facecolor='k', edgecolor='k', alpha=0.2, zorder=10) #green b2ffb2
        ax.add_collection(laser_1)
        laser_2 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(y_values)), ymax=1.3*(max(y_values)), where=jaaba_data['Laser2_state'] > 0, facecolor='r', edgecolor='r', alpha=0.1, zorder=11) #green b2ffb2
        ax.add_collection(laser_2)
       
        ax.set_xlim((np.amin(x_values),np.amax(x_values)))
        ax.set_ylim(0.85*(min(y_values)),1.15*(max(y_values)))
        ax.set_ylabel(measurements[m] , fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=16)
    return fig

def plot_data(means, sems, ns, measurement):
    fig = plt.figure()
    group_number = 0
    ax = fig.add_subplot(1,1,1)
    y_range = []
    for x in means.index.levels[0]:
        max_n = ns.ix[x]['FlyID'].max()
        x_values = []
        y_values = []
        psems = []
        nsems = []
        laser_x = []
        for w in means.ix[x].index:
            laser_x.append((w-pd.to_datetime(0)).total_seconds())
            if ns.ix[x]['FlyID'][w] >= ((max_n)/4): #(max_n/3):
                #print ns.ix[x]['FlyID'][w]
                x_values.append((w-pd.to_datetime(0)).total_seconds())
                y_values.append(means.ix[x,w][measurement])
                psems.append(sems.ix[x,w][measurement])
                nsems.append(-1.0*sems.ix[x,w][measurement])
        #x_values = list((means.ix[x].index - pd.to_datetime(0)).total_seconds())
        #y_values = list(means.ix[x][measurement])
        #psems = list(sems.ix[x][measurement])
        #nsems = list(-1*(sems.ix[x][measurement]))
        y_range.append(np.amin(y_values))
        y_range.append(np.amax(y_values))
        top_errbar = tuple(map(sum, zip(psems, y_values)))
        bottom_errbar = tuple(map(sum, zip(nsems, y_values)))
        p = plt.plot(x_values, y_values, linewidth=3, zorder=100,
                        linestyle = '-',
                        color=colourlist[group_number],
                        label=(x + ', n= ' + str(max_n))) 
        q = plt.fill_between(x_values, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.15, 
                            linewidth=0,
                            zorder=90,
                            color=colourlist[group_number],
                            )
        group_number += 1
    ax.set_xlim((np.amin(x_values),np.amax(x_values)))
    ax.set_ylim(0.85*(np.amin(y_range)),1.15*(np.amax(y_range)))
    if 'maxWingAngle' in measurement:
        ax.set_ylabel('Mean maximum wing angle (rad)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
    elif 'dtarget' in measurement:
        ax.set_ylabel('Mean min. distance to target (px)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
        
    else:
        ax.set_ylabel('Mean ' + measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Time (s)', fontsize=16)
    laser_1 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser1_state'] > 0.1, facecolor='#DCDCDC', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #green b2ffb2
    ax.add_collection(laser_1)
    laser_2 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser2_state'] > 0.1, facecolor='#FFB2B2', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #red FFB2B2
    ax.add_collection(laser_2)
    
    
    l = plt.legend()
    l.set_zorder(1000)
    if 'maxWingAngle' in measurement:
        fig.savefig(JAABA + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(JAABA + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(JAABA + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.png', bbox_inches='tight')
    else:
        fig.savefig(JAABA + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(JAABA + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(JAABA + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.png', bbox_inches='tight')

def plot_rasters(raw_pickle):
    df = pd.read_pickle(raw_pickle)
    grouped = df.groupby(['group','synced_time'])
    fig = plt.figure()
    group_number = 1
    for x in set(grouped.mean().index.levels[0]):
        ax = fig.add_subplot(1,len(set(grouped.mean().index.levels[0])), group_number)
        group_number +=1
        actions = []
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--jaabadir', type=str, required=True,
                            help='directory of JAABA csv files')  
    #parser.add_argument('--outputdir', type=str, required=True,
    #                        help='directory to store analysis')
    #parser.add_argument('--bagdir', type=str, required=True,
    #                        help='directory of bag files')
    parser.add_argument('--binsize', type=str, required=True,
                            help='integer and unit, such as "5s" or "4Min" or "500ms"')
    parser.add_argument('--experiment', type=str, required=False,
                            help='handle to select experiment from group (example: IRR-)')
    parser.add_argument('--threshold', type=str, required=False, default="0-1-0", 
                            help='list threshold boundaries and threshold, delimited by -. ex: 60-120-0.5   =   minimum 0.5 WEI between 60s and 120s.')
    parser.add_argument('--pool_controls', type=str,  required=False, default = "",
                            help="list exact strings of control genotypes delimited by comma ex: DB204-GP-IRR,DB202-GP-IRR")
    parser.add_argument('--pool_experiment', type=str,  required=False, default = '',
                            help="list exact strings of experimental genotypes delimited by comma ex: DB204-GP-IRR,DB202-GP-IRR")
    parser.add_argument('--compile_folders', type=str, required=False, default = False, 
                            help="Make True if you want to analyze data from copies of pickled data")
        
    args = parser.parse_args()

    JAABA = args.jaabadir
    HANDLE = args.experiment
    BAGS = JAABA + 'BAGS'
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = (args.threshold).split('-')
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = float(THRESH_ON), float(THRESH_OFF), float(WEI_THRESHOLD)
    POOL_CONTROL = [str(item) for item in args.pool_controls.split(',')]
    POOL_EXP = [str(item) for item in args.pool_experiment.split(',')]
    #OUTPUT = args.outputdir
    COMPILE_FOLDERS = args.compile_folders

    binsize = (args.binsize)
    print "BINSIZE: ", binsize
    colourlist = ['#333333','#0033CC',  '#AAAAAA','#0032FF','r','c','m','y', '#000000']

    #filename = '/tier2/dickson/bathd/FlyMAD/JAABA_tracking/140927/wing_angles_nano.csv'
    #binsize = '5s'  # ex: '1s' or '4Min' etc
    #BAG_FILE = '/groups/dickson/home/bathd/Dropbox/140927_flymad_rosbag_copy/rosbagOut_2014-09-27-14-53-54.bag'


    if COMPILE_FOLDERS == False:
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
        if not os.path.exists(JAABA + 'TRACES') ==True:
            os.makedirs(JAABA + 'TRACES')
            
        updated = False

        for directory in glob.glob(JAABA + '*' + HANDLE + '*' + '*zoom*'):
            FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
            if not os.path.exists(JAABA + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle') ==True:
                sync_jaaba_with_ros(directory)
                updated = True
                
        if updated == True:
            print 'Found unprocessed files for the chosen bin. Compiling data...'
            

    gather_data(glob.glob(JAABA + 'JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
    
    means, sems, ns = group_data(JAABA + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
        
        
        
    """
    if not os.path.exists(JAABA + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle') ==True:
        gather_data(glob.glob(JAABA + 'JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
        means, sems = group_data(JAABA + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')

    if not os.path.exists(JAABA + 'JAR/'+ HANDLE + '_mean_' + binsize + '.pickle') ==True:
        means, sems = group_data(JAABA + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    """
    means =  pd.read_pickle(JAABA + 'JAR/' + HANDLE + '_mean_' + binsize + '.pickle')
    sems = pd.read_pickle(JAABA + 'JAR/' + HANDLE + '_sem_' + binsize + '.pickle')
    ns = pd.read_pickle(JAABA + 'JAR/' + HANDLE + '_n_' + binsize + '.pickle')

    plot_data(means, sems, ns, 'maxWingAngle')    
    #plot_data(means, sems, ns, 'Length')
    #plot_data(means, sems, ns, 'Width')
    plot_data(means, sems, ns, 'dtarget')



