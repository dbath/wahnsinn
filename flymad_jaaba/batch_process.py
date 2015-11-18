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
from scipy.stats import kruskal
import flymad_jaaba.utilities as utilities
import flymad_jaaba.target_detector as target_detector
import flymad_jaaba.wing_detector as wing_detector
import flymad.fake_plotly as fake_plotly
import pprint
#import flymad.madplot as madplot


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

def calc_p_values(data, gt1_name, gt2_name,
                  stat_colname=None,
                  num_bins=50, bin_how='mean',
                  ):

    if stat_colname is None:
        raise ValueError("you must explicitly set stat_colname (try 'maxWingAngle')")
    
    data.index = data.index.astype(np.int64)  #LAZY DANNO. DROP TIMESTAMPS FOR BINNING.
    data['synced_ns'] = data.index
    
    df_ctrl = data[data.group == gt1_name][['FlyID', stat_colname, 'synced_ns']]
    df_exp = data[data.group == gt2_name][['FlyID', stat_colname, 'synced_ns']]

    align_start = df_ctrl.index.min()
    dalign = df_ctrl.index.max() - align_start

    p_values = DataFrame()

    if bin_how=='mean':
        bin_func = np.mean
    elif bin_how=='median':
        bin_func = np.median

    bins = np.linspace(0,dalign,num_bins+1) + align_start
    binned_ctrl = pd.cut(df_ctrl.index, bins, labels= bins[:-1])
    binned_exp = pd.cut(df_exp.index, bins, labels= bins[:-1])
    for x in binned_ctrl.levels:
        test1_full_dataset = df_ctrl[binned_ctrl == x]
        test2_full_dataset = df_exp[binned_exp == x]
        bin_start_time = test1_full_dataset['synced_ns'].min()
        bin_stop_time = test1_full_dataset['synced_ns'].max()

        test1 = []
        for obj_id, fly_group in test1_full_dataset.groupby('FlyID'):
            test1.append( bin_func(fly_group[stat_colname].values) )
        test1 = np.array(test1)
        
        test2 = []
        for obj_id, fly_group in test2_full_dataset.groupby('FlyID'):
            test2.append( bin_func(fly_group[stat_colname].values) )
        test2 = np.array(test2)
        
        try:
            hval, pval = kruskal(test1, test2)
        except ValueError as err:
            pval = 1.0

        dftemp = DataFrame({'Bin_number': x,
                            'P': pval,
                            'bin_start_time':bin_start_time,
                            'bin_stop_time':bin_stop_time,
                            'name1':gt1_name, 
                            'name2':gt2_name,
                            'test1_n':len(test1),
                            'test2_n':len(test2),
                            }, index=[x])
        p_values = pd.concat([p_values, dftemp])
    return p_values

def view_pairwise_stats( data, names, fig_prefix, **kwargs):
    """ data = output from gather_data().   
        names = list of groups (ex. ['foo','bar','baz'])
        fig_prefix = full path and filename (without extension) of plot name.
        **kwargs = 
    """
    pairs = []
    for i,name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if j<=i:
                continue
            pairs.append( (name1, name2 ) )

    graph_data = []
    layout=None
    pvalue_results = {}
    for pair in pairs:
        name1, name2 = pair
        pairwise_data = get_pairwise( data, name1, name2, **kwargs)
        if pairwise_data is not None:
            graph_data.append( pairwise_data['data'] )
            layout=pairwise_data['layout']
            pvalue_results[pair] = pairwise_data['df']

    if len( graph_data )==0:
        return

    result2 = fake_plotly.plot( graph_data, layout=layout)
    ax = result2['fig'].add_subplot(111)
    #ax.axhline( -np.log10(1), color='k', lw=0.2 )
    #ax.axhline( -np.log10(0.05), color='k', lw=0.2 )
    #ax.axhline( -np.log10(0.01), color='k', lw=0.2 )
    #ax.axhline( -np.log10(0.001), color='k', lw=0.2 )
    if len(graph_data)>=1:
        #only one pairwise comparison
        n_comparisons = len(pairwise_data['df'])
        ax.axhline( -np.log10(0.05/n_comparisons), color='r', lw=0.5, linestyle='--' )

    pprint.pprint(result2)
    for ext in ['.png','.svg']:
        fig_fname = fig_prefix + '_p_values' + ext
        result2['fig'].savefig(fig_fname)
        print 'saved',fig_fname

    return pvalue_results

def get_pairwise(data,gt1_name,gt2_name,**kwargs):
    layout_title = kwargs.pop('layout_title',None)
    #human_label_dict = kwargs.pop('human_label_dict',None)
    p_values = calc_p_values(data, gt1_name, gt2_name,**kwargs)
    if len(p_values)==0:
        return None

    starts = np.array(p_values['bin_start_time'].values)
    stops = np.array(p_values['bin_stop_time'].values)
    pvals = p_values['P'].values
    n1 = p_values['test1_n'].values
    n2 = p_values['test2_n'].values
    logs = -np.log10(pvals)

    xs = []
    ys = []
    texts = []

    for i in range(len(logs)):
        xs.append( starts[i] / 1000000000 ) #convert to seconds
        ys.append( logs[i] )
        texts.append( 'p=%.3g, n=%d,%d t=%s to %s'%(
            pvals[i], n1[i], n2[i], starts[i], stops[i] ) )

        xs.append( stops[i]  / 1000000000 ) #convert to seconds
        ys.append( logs[i] )
        texts.append( '')

    this_dict = {
        'name':'%s vs. %s' % (gt1_name, gt2_name),
        'x':[float(x) for x in xs],
        'y':[float(y) for y in ys],
        'text':texts,
        }

    layout = {
        'xaxis': {'title': 'Time (s)'},
        'yaxis': {'title': '-Log10(p)'},
        }
    if layout_title is not None:
        layout['title'] = layout_title
    results = {'data':this_dict,
               'layout':layout,
               'df':p_values,
               }
    return results
    
def sync_video_with_ros(FMF_DIR):

    print "Processing: ", FMF_DIR
    
        
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
    
    BAG_FILE                = match_fmf_and_bag(FMF_TIME)
    
    WIDE_FMF                = utilities.match_wide_to_zoom(FMF_DIR, MAIN_DIRECTORY+"/")
    
    targets = target_detector.TargetDetector(WIDE_FMF, FMF_DIR)
    targets.plot_targets_on_background()
    targets.plot_trajectory_on_background(BAG_FILE)    
    
    dtarget = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget']
    
    arenaCtr, Arena_radius = targets._arena.circ
    
    for x in utilities.find_files(FMF_DIR, '*zoom*.fmf'):
        ZOOM_FMF = x
    
    if not os.path.exists(MAIN_DIRECTORY + '/JAR/wing_data') ==True:
            os.makedirs(MAIN_DIRECTORY + '/JAR/wing_data')
    if args.save_wingext_pngs == True:
        if not os.path.exists(FMF_DIR + '/TRACKING_RESULTS'):
            os.makedirs(FMF_DIR + '/TRACKING_RESULTS')
        wingImageDir = FMF_DIR + '/TRACKING_RESULTS'
    else:
        wingImageDir=None
    
    
    if not os.path.exists(MAIN_DIRECTORY + '/JAR/wing_data/'+ FLY_ID + '_wing_angles.pickle') ==True:
        wings = wing_detector.WingDetector(ZOOM_FMF, BAG_FILE, dtarget, arenaCtr, wingImageDir )
        wings.execute()
        wings.wingData.to_pickle(MAIN_DIRECTORY + '/JAR/wing_data/'+ FLY_ID + '_wing_angles.pickle')
    
    videoData = pd.read_pickle(MAIN_DIRECTORY + '/JAR/wing_data/'+ FLY_ID + '_wing_angles.pickle')
    videoData[['Length','Width','Left','Right']] = videoData[['Length','Width','Left','Right']].astype(np.float64)
    videoData = convert_timestamps(videoData)

    # ALIGN LASER STATE DATA
    laser_states = utilities.get_laser_states(BAG_FILE)
    try:
        videoData['Laser0_state'] = laser_states['Laser0_state'].asof(videoData.index).fillna(value=1)
        videoData['Laser1_state'] = laser_states['Laser1_state'].asof(videoData.index).fillna(value=0)  #YAY! 
        videoData['Laser2_state'] = laser_states['Laser2_state'].asof(videoData.index).fillna(value=0)
    except:
        print "\t ERROR: problem interpreting laser current values."
        videoData['Laser0_state'] = 0
        videoData['Laser2_state'] = 0
        videoData['Laser1_state'] = 0
        
    
    # COMPUTE AND ALIGN DISTANCE TO NEAREST TARGET

    
    positions = utilities.get_positions_from_bag(BAG_FILE)
    positions = utilities.convert_timestamps(positions)
    videoData['fly_x'] = positions['fly_x'].asof(videoData.index).fillna(value=0)
    videoData['fly_y'] = positions['fly_y'].asof(videoData.index).fillna(value=0)
    
    
    videoData['dtarget'] = dTarget.asof(videoData.index).fillna(value=0)
    
    
    videoData['Timestamp'] = videoData.index #silly pandas bug for subtracting from datetimeindex...
    try:
        videoData['synced_time'] = videoData['Timestamp'] - videoData.Timestamp[(videoData.Laser2_state + videoData.Laser1_state) > 0.001].index[0]
    except:
        print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
        videoData['synced_time'] = videoData['Timestamp'] - videoData.Timestamp.index[0]
    videoData.index = videoData.synced_time
    videoData.index = pd.to_datetime(videoData.index)

    ###    WING EXTENSION    ###
    videoData['maxWingAngle'] = get_absmax(videoData[['Left','Right']])
    #videoData[videoData['maxWingAngle'] > 3.1] = np.nan
    
    program = 'dark' #FIXME
    

    
    if program == 'IRR':
        BEGINNING =videoData.Timestamp[videoData.synced_time >= -30000000000].index[0]#videoData.Timestamp.index[0]
        #FIRST_IR_ON = videoData.Timestamp[((videoData.Laser1_state > 0.001) & (videoData.synced_time >= -1))].index[0]
        FIRST_IR_ON = videoData.Timestamp[videoData.synced_time >= 0].index[0]
        #FIRST_IR_OFF = videoData.Timestamp[((videoData.Laser1_state > 0.001) & (videoData.synced_time <= 120))].index[-1]
        FIRST_IR_OFF = videoData.Timestamp[videoData.synced_time >= 60000000000].index[0]
        RED_ON = videoData.Timestamp[videoData.Laser2_state > 0.001].index[0]
        RED_OFF = videoData.Timestamp[videoData.Laser2_state > 0.001].index[-1]
        SECOND_IR_ON = videoData.Timestamp[videoData.synced_time >=320000000000].index[0]
        #SECOND_IR_ON = videoData.Timestamp[((videoData.Laser1_state > 0.001) & (videoData.synced_time >= 120))].index[0]
        SECOND_IR_OFF = videoData.Timestamp[videoData.Laser1_state > 0.001].index[-1]
        END = videoData.Timestamp.index[-1]
        
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, BEGINNING, FIRST_IR_ON, '1-prestim', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, FIRST_IR_ON, FIRST_IR_OFF, '2-IR1', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, FIRST_IR_OFF, RED_ON, '3-post-IR1', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, RED_ON,RED_OFF, '4-red', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE,RED_OFF, SECOND_IR_ON,'5-post-red', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE,SECOND_IR_ON,SECOND_IR_OFF,'6-IR2', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE,SECOND_IR_OFF,END,'7-post-IR2', background=False)
    
    
    
    if program == 'dark':
        BEGINNING =videoData.Timestamp[videoData.synced_time >= -30000000000].index[0]
        print set(videoData.Laser0_state), set(videoData.Laser1_state), set(videoData.Laser2_state)
        STIM_ON = videoData.Timestamp[videoData.Laser1_state > 0.001].index[0]
        STIM_OFF = videoData.Timestamp[videoData.Laser1_state > 0.001].index[-1]
        LIGHTS_OUT = videoData.Timestamp[videoData.Laser0_state < 0.001].index[0]
        LIGHTS_ON = videoData.Timestamp[videoData.Laser0_state < 0.001].index[-1]
        END = videoData.Timestamp.index[-1]
        
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, BEGINNING, STIM_ON, '1-prestim', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, STIM_ON,STIM_OFF, '2-stim', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, STIM_OFF, LIGHTS_OUT,'3-post-stim', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, LIGHTS_OUT, LIGHTS_ON,'4-DARK', background=False)
        targets.plot_trajectory_and_wingext(videoData, BAG_FILE, LIGHTS_ON, END,'7-light', background=False)
        
    
    targets.plot_trajectory_and_wingext(videoData, BAG_FILE)
    
    ### ABDOMINAL BENDING   ###
    videoData[videoData['Length'] > 110] = np.nan  #discard frames with bogus length.  *************
    videoData[videoData['Length'] < 60] = np.nan  #discard frames with bogus length.
    
    trace = plot_single_trace(videoData)
    trace.savefig(MAIN_DIRECTORY + '/TRACES/' + FLY_ID + '.png')
    plt.close('all')
    
    ###FIRST COURTSHIP BOUT AFTER STIMULUS###
    
    #courted, latency = latency_measures(videoData)
    
    
    if 'binsize' in globals():
        videoData = bin_data(videoData, binsize)
        videoData.to_pickle(MAIN_DIRECTORY + '/JAR/' + FLY_ID + '_' + binsize + '_fly.pickle')
    else:
        return videoData, courted, latency
    
def latency_measures(videoData): #takes input from unbinned videoData (during sync...)
      poststim = videoData[(videoData.index > videoData[(videoData.Laser2_state + videoData.Laser1_state) > 0.001].index[-1])]
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
        print x
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
    datadf.to_csv(MAIN_DIRECTORY +'/' + HANDLE + '_rawdata_' + binsize + '.csv', sep=',')
    datadf.to_pickle(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
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
    means.to_csv(MAIN_DIRECTORY + HANDLE + '_mean_' + binsize + '.csv')
    means.to_pickle(MAIN_DIRECTORY + '/JAR/'+HANDLE+ '_mean_' + binsize + '.pickle')
    ns.to_csv(MAIN_DIRECTORY + HANDLE + '_n_' + binsize + '.csv') 
    ns.to_pickle(MAIN_DIRECTORY + '/JAR/' + HANDLE + '_n_' + binsize + '.pickle')   
    sems.to_csv(MAIN_DIRECTORY + HANDLE + '_sem_' + binsize + '.csv')
    sems.to_pickle(MAIN_DIRECTORY + '/JAR/' + HANDLE + '_sem_' + binsize + '.pickle')   
    
    return means, sems, ns

def plot_single_trace(videoData):
    fig = plt.figure()
    measurements = ['maxWingAngle', 'Left','Right','dtarget']
    x_values=[]
    for w in videoData.index:
        x_values.append((w-pd.to_datetime(0)).total_seconds())
    
    for m in range(len(measurements)):
        ax = fig.add_subplot(len(measurements), 1,(m+1))
        y_values = videoData[measurements[m]]
        p = plt.plot(x_values, y_values, linewidth=2, zorder=100)
        
        if args.plot_ambient == True:
            laser_0 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(y_values.min()), ymax=1.3*(y_values.max()), where=videoData['Laser0_state'] == 0, facecolor='k', edgecolor='k', alpha=0.6, zorder=10) #green b2ffb2
            ax.add_collection(laser_0)
            
        laser_1 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(y_values.min()), ymax=1.3*(y_values.max()), where=videoData['Laser1_state'] > 0, facecolor='k', edgecolor='k', alpha=0.2, zorder=10) #green b2ffb2
        ax.add_collection(laser_1)
        
        laser_2 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(y_values)), ymax=1.3*(max(y_values)), where=videoData['Laser2_state'] > 0, facecolor='r', edgecolor='r', alpha=0.1, zorder=11) #green b2ffb2
        ax.add_collection(laser_2)
       
        ax.set_xlim((np.amin(x_values),np.amax(x_values)))
        ax.set_ylim(0.85*(np.amin(y_values)),1.15*(np.amax(y_values)))
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
            if ns.ix[x]['FlyID'][w] >= ((max_n)-2): #(max_n/3):
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
      
    if args.plot_ambient == True:
        laser_0 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=videoData['Laser0_state'] == 0, facecolor='#BBBBBB', linewidth=0, edgecolor=None, alpha=1.0, zorder=10)
        ax.add_collection(laser_0)
        
    laser_1 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser1_state'] > 0.1, facecolor='#DCDCDC', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #green b2ffb2
    ax.add_collection(laser_1)
    laser_2 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser2_state'] > 0.1, facecolor='#FFB2B2', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #red FFB2B2
    ax.add_collection(laser_2)
    
    
    l = plt.legend()
    l.set_zorder(1000)
    if 'maxWingAngle' in measurement:
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.png', bbox_inches='tight')
    else:
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(MAIN_DIRECTORY +'/' + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.png', bbox_inches='tight')

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
    parser.add_argument('--rawdatadir', type=str, required=True,
                            help='directory of FMF and bag files')  
    #parser.add_argument('--outputdir', type=str, required=True,
    #                        help='directory to store analysis')
    #parser.add_argument('--bagdir', type=str, required=True,
    #                        help='directory of bag files')
    
    parser.add_argument('--save_wing_detection_dir', type=str, required=False, default=None,
                            help='full path to directory where wing angle png files should be stored. example: /full/path')
    parser.add_argument('--binsize', type=str, required=True,
                            help='integer and unit, such as "5s" or "4Min" or "500ms"')
    parser.add_argument('--experiment', type=str, required=False,
                            help='handle to select experiment from group (example: IRR-)')
    parser.add_argument('--threshold', type=str, required=False, default="0-100-0", 
                            help='list threshold boundaries and threshold, delimited by -. ex: 60-120-0.5   =   minimum 0.5 WEI between 60s and 120s.')
    parser.add_argument('--pool_controls', type=str,  required=False, default = "",
                            help="list exact strings of control genotypes delimited by comma ex: DB204-GP-IRR,DB202-GP-IRR")
    parser.add_argument('--pool_experiment', type=str,  required=False, default = '',
                            help="list exact strings of experimental genotypes delimited by comma ex: DB204-GP-IRR,DB202-GP-IRR")
    parser.add_argument('--compile_folders', type=bool, required=False, default = False, 
                            help="Make True if you want to analyze data from copies of pickled data")
    parser.add_argument('--save_wingext_pngs', type=bool, required=False, default = False, 
                            help="Make 1 if you want to save results of wing detection")
    parser.add_argument('--plot_ambient', type=str, required=False, default = False, 
                            help="Make True if you want to plot the absence of ambient light from laser0.")
        
    args = parser.parse_args()

    MAIN_DIRECTORY = args.rawdatadir
    HANDLE = args.experiment
    BAGS = MAIN_DIRECTORY + '/BAGS'
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = (args.threshold).split('-')
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = float(THRESH_ON), float(THRESH_OFF), float(WEI_THRESHOLD)
    POOL_CONTROL = [str(item) for item in args.pool_controls.split(',')]
    POOL_EXP = [str(item) for item in args.pool_experiment.split(',')]
    #OUTPUT = args.outputdir
    COMPILE_FOLDERS = args.compile_folders

    binsize = (args.binsize)
    print "BINSIZE: ", binsize
    colourlist = ['#333333','#0033CC',  '#AAAAAA','#0032FF','r','c','m','y', '#000000']




    if 1:#COMPILE_FOLDERS == False:
        baglist = []
        for bag in glob.glob(BAGS + '/*.bag'):
            bagtimestamp = parse_bagtime(bag)
            baglist.append((bag, bagtimestamp))
        bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
        bagframe.index = pd.to_datetime(bagframe['Timestamp'])
        bagframe = bagframe.sort()
        bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')

        if not os.path.exists(MAIN_DIRECTORY + '/JAR') ==True:
            print "MAKING A JAR"
            os.makedirs(MAIN_DIRECTORY + '/JAR')
        if not os.path.exists(MAIN_DIRECTORY + '/TRACES') ==True:
            os.makedirs(MAIN_DIRECTORY + '/TRACES')
        if not os.path.exists(MAIN_DIRECTORY + '/JAR/wing_data') ==True:
                os.makedirs(MAIN_DIRECTORY + '/JAR/wing_data')            
        updated = False

        for directory in glob.glob(MAIN_DIRECTORY + '/*' + HANDLE + '*' + '*zoom*'):
            FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
            if not os.path.exists(MAIN_DIRECTORY + '/JAR/' + FLY_ID + '_' + binsize + '_fly.pickle') ==True:
                sync_video_with_ros(directory)
                updated = True
                
        if updated == True:
            print 'Found unprocessed files for the chosen bin. Compiling data...'
            

    gather_data(glob.glob(MAIN_DIRECTORY + '/JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
    
    means, sems, ns = group_data(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
        
        
        
    """
    if not os.path.exists(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle') ==True:
        gather_data(glob.glob(MAIN_DIRECTORY + '/JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
        means, sems = group_data(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')

    if not os.path.exists(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_mean_' + binsize + '.pickle') ==True:
        means, sems = group_data(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    """
    means =  pd.read_pickle(MAIN_DIRECTORY + '/JAR/' + HANDLE + '_mean_' + binsize + '.pickle')
    sems = pd.read_pickle(MAIN_DIRECTORY + '/JAR/' + HANDLE + '_sem_' + binsize + '.pickle')
    ns = pd.read_pickle(MAIN_DIRECTORY + '/JAR/' + HANDLE + '_n_' + binsize + '.pickle')


    plot_these = ['maxWingAngle','dtarget']
    
    rawdata = pd.read_pickle(MAIN_DIRECTORY + '/JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    rawdata = pool_genotypes(rawdata)
    for measurement in plot_these:
        plot_data(means, sems, ns, measurement)    
        fname_prefix = MAIN_DIRECTORY + HANDLE + '_p-values_' + measurement + '_' + binsize + '_bins'
        pp = view_pairwise_stats(rawdata, list(set(rawdata.group)), fname_prefix,
                                       stat_colname=measurement,
                                       layout_title=('Kruskal-Wallis H-test: ' + measurement),
                                       num_bins=len(set(rawdata.index))/20,
                                       )



