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
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
from multiprocessing import Process
import traceback

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
    
def sync_jaaba_with_ros(FMF_DIR):

    print "Processing: ", FMF_DIR
    
    DATADIR_CSV               = FMF_DIR + '/registered_trx.csv'
    
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(FMF_DIR)
    
    BAG_FILE                = match_fmf_and_bag(FMF_TIME)
    
    WIDE_FMF                = utilities.match_wide_to_zoom(FMF_DIR, DATADIR)
    
    
    
    for x in glob.glob(FMF_DIR +'/*zoom*'+'*.fmf'):
        ZOOM_FMF = x

    
    if not os.path.exists(FMF_DIR + '/TRACKING_FRAMES') == True:
        os.makedirs(FMF_DIR + '/TRACKING_FRAMES')
    
    # COMPUTE AND ALIGN DISTANCE TO NEAREST TARGET
    targets = target_detector.TargetDetector(WIDE_FMF, FMF_DIR)
    targets.plot_targets_on_background()
    targets.plot_trajectory_on_background(BAG_FILE)
    
    dtarget = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget']
    (arena_centre), arena_radius = targets._arena.circ
    
    if MAKE_MOVIES:
        TRACKING_DIRECTORY = FMF_DIR + '/TRACKING_FRAMES/'
    else:
        TRACKING_DIRECTORY = None
    
    if os.path.exists(FMF_DIR + '/wingdata.pickle'):
        datadf = pd.read_pickle(FMF_DIR + '/wingdata.pickle')
        datadf.columns= ['BodyAxis','leftAngle','leftWingLength','Length','rightAngle','rightWingLength','target_angle_TTM',
                                     'target_distance_TTM','Timestamp','Width']
    else:
        try:
            wings = wing_detector.WingDetector(ZOOM_FMF, BAG_FILE, dtarget, arena_centre, TRACKING_DIRECTORY )
            
            wings.execute()
            wings.wingData.columns= ['BodyAxis','leftAngle','leftWingLength','Length','rightAngle','rightWingLength','target_angle_TTM',
                                     'target_distance_TTM','Timestamp','Width']
            wings.wingData.to_pickle(FMF_DIR + '/wingdata.pickle')
            wings.tracking_info.to_pickle(FMF_DIR + '/tracking_info.pickle')
            datadf = DataFrame(wings.wingData)
        
        
            if MAKE_MOVIES:
                utilities.call_command("ffmpeg -f image2 -r 15 -i "+ TRACKING_DIRECTORY + "_tmp%05d.png -vf scale=iw/2:-1 -vcodec mpeg4 -b 8000k -y " + FMF_DIR + "/tracked_movie.mp4;")
                utilities.call_command("rm -r " + TRACKING_DIRECTORY)
                #wings.make_movie(wings._tempdir, wings.fmf_file.rsplit('/',1)[0]+'/tracked_movie.mp4',15)
            
        except Exception,e:
            traceback.print_exc() 
            print 'ERROR PROCESSING WING TRACKING...', FMF_DIR
            print str(e)
            
            return
            
    datadf['Frame_number'] = datadf.index
    datadf = convert_timestamps(datadf)

    # ALIGN LASER STATE DATA
    laser_states = utilities.get_laser_states(BAG_FILE)
    try:
        datadf['Laser0_state'] = laser_states['Laser0_state'].asof(datadf.index).fillna(value=1)
        datadf['Laser1_state'] = laser_states['Laser1_state'].asof(datadf.index).fillna(value=0)  #YAY! 
        datadf['Laser2_state'] = laser_states['Laser2_state'].asof(datadf.index).fillna(value=0)
    except:
        print "\t ERROR: problem interpreting laser current values."
        datadf['Laser0_state'] = 0
        datadf['Laser2_state'] = 0
        datadf['Laser1_state'] = 0
        

    
    positions = utilities.get_positions_from_bag(BAG_FILE)
    positions = utilities.convert_timestamps(positions)
    datadf['fly_x'] = positions['fly_x'].asof(datadf.index).fillna(method='pad')
    datadf['fly_y'] = positions['fly_y'].asof(datadf.index).fillna(method='pad')
    
    datadf['dcentre'] = np.sqrt(((datadf['fly_x']-arena_centre[0])/5.2)**2 + ((datadf['fly_y']-arena_centre[1])/5.2)**2)
    dtarget_temp = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget'].asof(datadf.index).fillna(method='pad')
    datadf['dtarget'] = datadf['target_distance_TTM'].fillna(dtarget_temp)
    
    
    datadf['Timestamp'] = datadf.index #silly pandas bug for subtracting from datetimeindex...
    
    number_of_bouts, bout_duration, first_TS, last_TS = utilities.detect_stim_bouts(datadf, 'Laser1_state')
    
    try:
        datadf['synced_time'] = datadf['Timestamp'] - datadf.Timestamp[(datadf.Laser2_state + datadf.Laser1_state) > 0.001].index[0]
    except:
        print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
        datadf['synced_time'] = datadf['Timestamp'] - datadf.Timestamp.index[0]
    datadf.index = datadf.synced_time
    datadf.index = pd.to_datetime(datadf.index)

    ###    WING EXTENSION    ###
    datadf['maxWingAngle'] = get_absmax(datadf[['leftAngle','rightAngle']]).fillna(method='pad')
    datadf['maxWingLength'] = get_absmax(datadf[['leftWingLength','rightWingLength']]).fillna(method='pad')
    #datadf[datadf['maxWingAngle'] > 3.1] = np.nan

    number_of_bouts, stim_duration, first_TS, last_TS = utilities.detect_stim_bouts(datadf, 'Laser2_state')  #HACK DANNO
    datadf['stim_duration'] = stim_duration



    """
    program = 'dark'
    
    plt.plot(datadf.Timestamp, datadf.Laser0_state, 'b')
    plt.plot(datadf.Timestamp, datadf.Laser1_state, 'k')
    plt.plot(datadf.Timestamp, datadf.Laser2_state, 'r')
    plt.show()
    
    if program == 'IRR':
        BEGINNING =datadf.Timestamp[datadf.synced_time >= -30000000000].index[0]#datadf.Timestamp.index[0]
        #FIRST_IR_ON = datadf.Timestamp[((datadf.Laser1_state > 0.001) & (datadf.synced_time >= -1))].index[0]
        FIRST_IR_ON = datadf.Timestamp[datadf.synced_time >= 0].index[0]
        #FIRST_IR_OFF = datadf.Timestamp[((datadf.Laser1_state > 0.001) & (datadf.synced_time <= 120))].index[-1]
        FIRST_IR_OFF = datadf.Timestamp[datadf.synced_time >= 60000000000].index[0]
        RED_ON = datadf.Timestamp[datadf.Laser2_state > 0.001].index[0]
        RED_OFF = datadf.Timestamp[datadf.Laser2_state > 0.001].index[-1]
        SECOND_IR_ON = datadf.Timestamp[datadf.synced_time >=320000000000].index[0]
        #SECOND_IR_ON = datadf.Timestamp[((datadf.Laser1_state > 0.001) & (datadf.synced_time >= 120))].index[0]
        SECOND_IR_OFF = datadf.Timestamp[datadf.Laser1_state > 0.001].index[-1]
        END = datadf.Timestamp.index[-1]
        
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, BEGINNING, FIRST_IR_ON, '1-prestim', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, FIRST_IR_ON, FIRST_IR_OFF, '2-IR1', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, FIRST_IR_OFF, RED_ON, '3-post-IR1', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, RED_ON,RED_OFF, '4-red', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE,RED_OFF, SECOND_IR_ON,'5-post-red', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE,SECOND_IR_ON,SECOND_IR_OFF,'6-IR2', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE,SECOND_IR_OFF,END,'7-post-IR2', background=False)
    
    
    
    if program == 'dark':
        BEGINNING =datadf.Timestamp[datadf.synced_time >= -30000000000].index[0]
        print set(datadf.Laser0_state), set(datadf.Laser1_state), set(datadf.Laser2_state)
        STIM_ON = datadf.Timestamp[datadf.Laser1_state > 0.001].index[0]
        STIM_OFF = datadf.Timestamp[datadf.Laser1_state > 0.001].index[-1]
        LIGHTS_OUT = datadf.Timestamp[datadf.Laser0_state < 0.001].index[0]
        LIGHTS_ON = datadf.Timestamp[datadf.Laser0_state < 0.001].index[-1]
        END = datadf.Timestamp.index[-1]
        
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, BEGINNING, STIM_ON, '1-prestim', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, STIM_ON,STIM_OFF, '2-stim', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, STIM_OFF, LIGHTS_OUT,'3-post-stim', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, LIGHTS_OUT, LIGHTS_ON,'4-DARK', background=False)
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE, LIGHTS_ON, END,'7-light', background=False)
        
    """
    
    try:
        targets.plot_trajectory_and_wingext(datadf, BAG_FILE)
    except Exception,e:
            traceback.print_exc() 
            print 'ERROR generating targeting plot:', FMF_DIR
            print str(e)
            
    ### ABDOMINAL BENDING   ###
    #datadf[datadf['Length'] > 110] = np.nan  #discard frames with bogus length.  *************
    #datadf[datadf['Length'] < 60] = np.nan  #discard frames with bogus length.
    
    trace = plot_single_trace(datadf)
    trace.savefig(DATADIR + 'TRACES/' + FLY_ID + '.png')
    plt.close('all')
    
    ###FIRST COURTSHIP BOUT AFTER STIMULUS###
    
    #courted, latency = latency_measures(datadf)
    
    datadf.to_pickle(FMF_DIR + '/frame_by_frame_synced.pickle')
    datadf.to_csv(FMF_DIR + '/frame_by_frame_synced.csv', sep=',')
    
    if 'binsize' in globals():
        datadf = bin_data(datadf, binsize)
        datadf.to_pickle(DATADIR + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle')
    else:
        return datadf, courted, latency
    
def latency_measures(datadf): #takes input from unbinned datadf (during sync...)
      poststim = datadf[(datadf.index > datadf[(datadf.Laser2_state + datadf.Laser1_state) > 0.001].index[-1])]
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

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]     
    
def gather_data(filelist):
    datadf = DataFrame()
    intvals = np.array([0, 200, 2000, 20000]) #6310
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
        if DOSE_RESPONSE:
            try:
                number_of_bouts, bout_duration, first_TS, last_TS = utilities.detect_stim_bouts(fx, 'Laser2_state')
            except:
                number_of_bouts = 1
            
            stim_duration = find_nearest(intvals, fx['stim_duration'][0])
            fx['group'] = str(number_of_bouts) + 'x_' + str(stim_duration) + 'ms'
        else:
            fx['group'] = EXP_ID
        fx['FlyID'] = FLY_ID
        datadf = pd.concat([datadf, fx])
    datadf.to_csv(DATADIR + HANDLE + '_rawdata_' + binsize + '.csv', sep=',')
    datadf.to_pickle(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
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
    means.to_csv(DATADIR + HANDLE + '_mean_' + binsize + '.csv')
    means.to_pickle(DATADIR + 'JAR/'+HANDLE+ '_mean_' + binsize + '.pickle')
    ns.to_csv(DATADIR + HANDLE + '_n_' + binsize + '.csv') 
    ns.to_pickle(DATADIR + 'JAR/' + HANDLE + '_n_' + binsize + '.pickle')   
    sems.to_csv(DATADIR + HANDLE + '_sem_' + binsize + '.csv')
    sems.to_pickle(DATADIR + 'JAR/' + HANDLE + '_sem_' + binsize + '.pickle')   
    
    return means, sems, ns

def plot_single_trace(datadf):
    fig = plt.figure()
    measurements = ['maxWingAngle', 'dtarget', 'Length']
    x_values=[]
    for w in datadf.index:
        x_values.append((w-pd.to_datetime(0)).total_seconds())
    
    for m in range(len(measurements)):
        ax = fig.add_subplot(len(measurements), 1,(m+1))
        y_values = datadf[measurements[m]]
        p = plt.plot(x_values, y_values, linewidth=2, zorder=100)
        
        if args.plot_ambient == True:
            laser_0 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(y_values.min()), ymax=1.3*(y_values.max()), where=datadf['Laser0_state'] == 0, facecolor='k', edgecolor='k', alpha=0.6, zorder=10) #green b2ffb2
            ax.add_collection(laser_0)
            
        laser_1 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(y_values.min()), ymax=1.3*(y_values.max()), where=datadf['Laser1_state'] > 0, facecolor='k', edgecolor='k', alpha=0.2, zorder=10) #green b2ffb2
        ax.add_collection(laser_1)
        
        laser_2 = collections.BrokenBarHCollection.span_where(x_values, ymin=0.85*(min(y_values)), ymax=1.3*(max(y_values)), where=datadf['Laser2_state'] > 0, facecolor='r', edgecolor='r', alpha=0.1, zorder=11) #green b2ffb2
        ax.add_collection(laser_2)
       
        ax.set_xlim((np.amin(x_values),np.amax(x_values)))
        ax.set_ylim(0.85*(np.amin(y_values)),1.15*(np.amax(y_values)))
        ax.set_ylabel(measurements[m] , fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=16)
    return fig

def plot_data(means, sems, ns, measurement):
    means = means[means[measurement].notnull()]
    #ns = ns[ns[measurement].notnull()]
    #sems = sems[sems[measurement].notnull()]
    fig = plt.figure()
    group_number = 0
    ax = fig.add_subplot(1,1,1)
    y_range = []
    x_range = []
    laser_x = []
    for x in means.index.levels[0]:
        max_n = ns.ix[x]['FlyID'].max()
        x_values = []
        y_values = []
        psems = []
        nsems = []
        for w in means.ix[x].index:
            laser_x.append((w-pd.to_datetime(0)).total_seconds())
            if ns.ix[x]['FlyID'][w] >= ((max_n)/3): #(max_n/3):
                x_range.append((w-pd.to_datetime(0)).total_seconds())
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
    if 'maxWingAngle' in measurement:
        ax.set_ylabel('Mean maximum wing angle (rad)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
    elif 'dtarget' in measurement:
        ax.set_ylabel('Mean min. distance to target (px)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
        
    else:
        ax.set_ylabel('Mean ' + measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Time (s)', fontsize=16)      
    
    #laser_x =  means.index.levels[1][(means.index.levels[1] > pd.to_datetime(np.amin(x_range)*1e9)) & (means.index.levels[1] < pd.to_datetime(np.amax(x_range)*1e9))]
     
    if args.plot_ambient == True:
        laser_0 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser0_state'] == 0, facecolor='#BBBBBB', linewidth=0, edgecolor=None, alpha=1.0, zorder=10)
        ax.add_collection(laser_0)
    
        
    laser_1 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser1_state'] > 0.1, facecolor='#DCDCDC', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #green b2ffb2
    ax.add_collection(laser_1)
    laser_2 = collections.BrokenBarHCollection.span_where(laser_x, ymin=0.85*(means[measurement].min()), ymax=1.15*(means[measurement].max()), where=means['Laser2_state'] > 0.1, facecolor='#FFB2B2', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #red FFB2B2
    ax.add_collection(laser_2)
    ax.set_xlim((np.amin(x_range),np.amax(x_range)))
    ax.set_ylim(0.85*(np.amin(y_range)),1.15*(np.amax(y_range)))
    
    
    l = plt.legend()
    l.set_zorder(1000)
    if 'maxWingAngle' in measurement:
        fig.savefig(DATADIR + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(DATADIR + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(DATADIR + HANDLE + '_mean_max_wing_angle_' + binsize + '_bins.png', bbox_inches='tight')
    else:
        fig.savefig(DATADIR + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.svg', bbox_inches='tight')
        fig.savefig(DATADIR + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.pdf', bbox_inches='tight')
        fig.savefig(DATADIR + HANDLE + '_mean_' + measurement + '_' + binsize + '_bins.png', bbox_inches='tight')

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
                            help='directory of fmf and bag files')  
    #parser.add_argument('--outputdir', type=str, required=True,
    #                        help='directory to store analysis')
    #parser.add_argument('--bagdir', type=str, required=True,
    #                        help='directory of bag files')
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
    parser.add_argument('--compile_folders', type=str, required=False, default = False, 
                            help="Make True if you want to analyze data from copies of pickled data")
    parser.add_argument('--plot_ambient', type=str, required=False, default = False, 
                            help="Make True if you want to plot the absence of ambient light from laser0.")
    parser.add_argument('--make_tracking_movie', type=str, required=False, default = False, 
                            help="Make True if you want to save annotated frames for tracking movies.")
    parser.add_argument('--dose_response', type=str, required=False, default = False, 
                            help="Make True if you want to group by stimulus paradigm.")
    
        
    args = parser.parse_args()

    DATADIR = args.rawdatadir
    if not DATADIR[-1] == '/' :
        DATADIR = DATADIR + '/'
    HANDLE = args.experiment
    MAKE_MOVIES = args.make_tracking_movie
    BAGS = DATADIR + 'BAGS'
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = (args.threshold).split('-')
    THRESH_ON, THRESH_OFF, WEI_THRESHOLD = float(THRESH_ON), float(THRESH_OFF), float(WEI_THRESHOLD)
    POOL_CONTROL = [str(item) for item in args.pool_controls.split(',')]
    POOL_EXP = [str(item) for item in args.pool_experiment.split(',')]
    #OUTPUT = args.outputdir
    COMPILE_FOLDERS = args.compile_folders
    DOSE_RESPONSE = args.dose_response

    binsize = (args.binsize)
    print "BINSIZE: ", binsize
    colourlist = ['#333333','#0033CC', '#AAAAAA','#6699FF', '#202020','#0032FF','r','c','m','y', '#000000']

    #filename = '/tier2/dickson/bathd/FlyMAD/DATADIR_tracking/140927/wing_angles_nano.csv'
    #binsize = '5s'  # ex: '1s' or '4Min' etc
    #BAG_FILE = '/groups/dickson/home/bathd/Dropbox/140927_flymad_rosbag_copy/rosbagOut_2014-09-27-14-53-54.bag'


    if not COMPILE_FOLDERS:
        baglist = []
        for bag in glob.glob(BAGS + '/*.bag'):
            bagtimestamp = parse_bagtime(bag)
            baglist.append((bag, bagtimestamp))
        bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
        bagframe.index = pd.to_datetime(bagframe['Timestamp'])
        bagframe = bagframe.sort()
        bagframe.to_csv(BAGS + '/list_of_bags.csv', sep=',')

        if not os.path.exists(DATADIR + 'JAR') ==True:
            print "MAKING A JAR"
            os.makedirs(DATADIR + 'JAR')
        if not os.path.exists(DATADIR + 'TRACES') ==True:
            os.makedirs(DATADIR + 'TRACES')
            
        updated = False
    threadcount = 0
    _filelist = []
    for _directory in glob.glob(DATADIR + '*' + HANDLE + '*' + '*zoom*'):
        _filelist.append(_directory)
    for directory in np.arange(len(_filelist)):    
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(_filelist[directory])
        if not os.path.exists(DATADIR + 'JAR/' + FLY_ID + '_' + binsize + '_fly.pickle') ==True:
            p = Process(target=sync_jaaba_with_ros, args=(_filelist[directory],))
            p.start()
            threadcount +=1
            
            if p.is_alive():
                if threadcount >=8:
                    threadcount = 0
                    p.join()
                elif _filelist[directory] == _filelist[-1]:
                    threadcount=0
                    p.join()

            

    gather_data(glob.glob(DATADIR + 'JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
    
    means, sems, ns = group_data(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
        
        
        
    """
    if not os.path.exists(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle') ==True:
        gather_data(glob.glob(DATADIR + 'JAR/*' + HANDLE + '*' + binsize + '_fly.pickle'))
        means, sems = group_data(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')

    if not os.path.exists(DATADIR + 'JAR/'+ HANDLE + '_mean_' + binsize + '.pickle') ==True:
        means, sems = group_data(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    """
    means =  pd.read_pickle(DATADIR + 'JAR/' + HANDLE + '_mean_' + binsize + '.pickle')
    sems = pd.read_pickle(DATADIR + 'JAR/' + HANDLE + '_sem_' + binsize + '.pickle')
    ns = pd.read_pickle(DATADIR + 'JAR/' + HANDLE + '_n_' + binsize + '.pickle')


    plot_these = ['maxWingAngle','dtarget', 'Length']
    
    rawdata = pd.read_pickle(DATADIR + 'JAR/'+ HANDLE + '_rawdata_' + binsize + '.pickle')
    rawdata = pool_genotypes(rawdata)
    for measurement in plot_these:
        plot_data(means, sems, ns, measurement)    
        fname_prefix = DATADIR + HANDLE + '_p-values_' + measurement + '_' + binsize + '_bins'
        pp = view_pairwise_stats(rawdata, list(set(rawdata.group)), fname_prefix,
                                       stat_colname=measurement,
                                       layout_title=('Kruskal-Wallis H-test: ' + measurement),
                                       num_bins=len(set(rawdata.index))/50,
                                       )




