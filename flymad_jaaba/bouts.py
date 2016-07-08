from scipy.stats import ttest_ind
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse
from pykalman import UnscentedKalmanFilter




ukf = UnscentedKalmanFilter(initial_state_mean=0, n_dim_obs=1)


def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def generate_all_pairs(list_of_names):
    pairs = []
    for i,name1 in enumerate(list_of_names):
        for j, name2 in enumerate(list_of_names):
            if j<=i:
                continue
            pairs.append( (name1, name2 ) )
    return pairs
    
def do_stats(pairs, dataset, group_col, stat_col, time_period):
    pvalue_results = {}
    for pair in pairs:
        name1, name2 = pair
        g1 = dataset[dataset[group_col] == name1]
        g2 = dataset[dataset[group_col] == name2]
        pvalue_results[pair] = np.around(ttest_ind(g1[stat_col], g2[stat_col])[1], 5)
    s = pd.Series(pvalue_results)
    df = pd.DataFrame(s.reset_index())
    
    df.columns = ['group1','group2','p-value']
    df['time_period'] = time_period
    return df  

colourlist = ['#8000CC','#2020CC'] 

def smooth(df, colname):
    df[colname + '_smoothed'] = ukf.smooth(df[colname].values)[0]
    return df  

def barplot(grouped_df, _column, statistic):
    means = grouped_df.groupby(level=[0,1]).mean()
    #set_of_groups = set(grouped_df['GROUP']))
    bar_width = 1.0/(len(means.index))
    error_config = {'ecolor': '0.1'}
    #if 'Index' in statistic:
    sems = grouped_df.groupby(level=[0,1]).sem().fillna(0)
    #else:
    #    sems = grouped_df.groupby(level=[0,1]).std().fillna(0)  #USE STANDARD DEVIATION, NOT SEM.
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    
    plt.bar(np.arange(0.1,(len(means.index)+0.1),1), 
            means[_column].fillna(0), 
            color= '#AAAAAA',
            yerr=sems[_column].fillna(0),
            error_kw=error_config,
            label=list(means.index))
    
    #ax.set_ylim(0,1.3*(means[_column].values.max()))
    
    if means[_column].values.min() >= 0:
        ax.set_ylim(0,1.1*((means[_column] + sems[_column]).values.max()))
    else:
        ax.set_ylim(1.1*((means[_column] - sems[_column]).values.min()),1.1*((means[_column] + sems[_column]).values.max()))
    
    #if 'Index' in statistic:
    ax.set_ylabel(statistic + ' ' + args.parameter + ' ' + u"\u00B1" + ' SEM', fontsize=20)   # +/- sign is u"\u00B1"
    #else:
    #    ax.set_ylabel(statistic + ' ' + args.parameter + ' ' + u"\u00B1" + ' SD', fontsize=20)
    ax.set_xticks(np.arange(0.1+bar_width/2.0,(len(means.index)+0.1+(bar_width/2.0)),1)) # (bar_width*(len(list_of_genotypes)/2)))
    ax.set_xticklabels(list(means.index), rotation=90)
    ax.tick_params(axis='y', labelsize=16 )
    ax.set_xlabel('Genotype', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    #plt.legend(prop={'size':10})           
    plt.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) + '_'+statistic.replace(' ','_')+'.svg', bbox_inches='tight')
    return

def lineplot(grouped_df, _column, statistic):
    meansALL = grouped_df.groupby(level=[0,1]).mean()
    if 'Max' in statistic:
        semsALL = grouped_df.groupby(level=[0,1]).std()
    else:
        semsALL = grouped_df.groupby(level=[0,1]).sem()
    #set_of_groups = set(grouped_df['GROUP']))
    error_config = {'ecolor': '0.1'}
    
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    
    adj = np.linspace(-0.025,0.025, len(grouped_df.index.levels[0]))
    
    for x in range(len(grouped_df.index.levels[0])):
        category = meansALL.index.levels[0][x]
        treatments = meansALL.index.levels[1]
        means = meansALL.ix[category]
        xs = np.arange(0, len(means.index),1) + adj[x]
        bar_width = 1.0/(len(means.index))
        sems = semsALL.ix[category]
    
        ax.errorbar(xs, 
                means[_column].fillna(0), 
                color= colourlist[x],
                yerr=sems[_column].fillna(0),
                fmt='--o',
                ecolor='#202020',
                label=category)
    
    #ax.set_ylim(0,1.3*(means[_column].values.max()))
    
    if meansALL[_column].values.min() >= 0:
        ax.set_ylim(0,1.1*((meansALL[_column] + semsALL[_column]).values.max()))
    else:
        ax.set_ylim(1.1*((meansALL[_column] - semsALL[_column]).values.min()),1.1*((meansALL[_column] + semsALL[_column]).values.max()))
    
    ax.margins(0.2)
    if 'Max' in statistic:
        ax.set_ylabel(statistic + ' ' + args.parameter + ' ' + u"\u00B1" + ' SD', fontsize=20)   # +/- sign is u"\u00B1"
    else:
        ax.set_ylabel(statistic + ' ' + args.parameter + ' ' + u"\u00B1" + ' SEM', fontsize=20)
    ax.set_xticks(np.arange(0,len(means.index),1))
    ax.set_xticklabels(list(means.index), rotation=90)
    ax.tick_params(axis='y', labelsize=16 )
    ax.set_xlabel('Stimulus', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        
    plt.legend(prop={'size':10})           
    plt.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) + '_'+statistic.replace(' ','_')+'_line.svg', bbox_inches='tight')
    return

def boutlength_distribution(df, _bins=np.arange(0,10,100)):
    dist = np.hist(df['bout_duration'], bins=_bins)[0]
    count = sum(dist)
    return dist/float(count)

def timeplot(grouped_df, measurement, statistic):
        
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    group_number = 0
    for x in grouped_df.groupby(level=0):
        mean = pd.Series(x[1].groupby(level=[0,2]).mean()[measurement])
        if 'duration' in statistic:
            sem  = pd.Series(x[1].groupby(level=[0,2]).sem()[measurement])
        else:
            sem  = pd.Series(x[1].groupby(level=[0,2]).std()[measurement])
        n = len(x[1].groupby(level=[0,1]).count())
               
        plt.plot(pd.Series(mean.index.levels[1].values),
                mean,
                linewidth=3, zorder=100,
                linestyle = '-',
                color=colourlist[group_number],
                label=(x[0] + ', n= ' + str(n)))
        plt.fill_between(pd.Series(mean.index.levels[1].values),
                        mean+sem, 
                        mean-sem, 
                        alpha=0.15, 
                        linewidth=0,
                        zorder=90,
                        color=colourlist[group_number],
                        )
        group_number +=1
    if 'duration' in statistic:
        ax.set_ylabel(statistic+' ' + measurement  + u"\u00B1" + ' SEM', fontsize=16)
    else:
        ax.set_ylabel(statistic+' ' + measurement  + u"\u00B1" + ' SD', fontsize=16)
        
    ax.set_xlabel('Minutes since stimulus onset', fontsize=16)    
    #ax.set_xlim((np.amin(),np.amax(x_range)))
    #ax.set_ylim(0.85*(np.amin(grouped_df.mean()[measurement]),1.15*(np.amax(grouped_df.mean()[measurement]))))
    l = plt.legend()
    l.set_zorder(1000)
    fig.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) +'_'+statistic+ '_'+ measurement +'_vs_time.svg', bbox_inches='tight')
    
def get_bouts(_fbf, parameter, threshold, time_period, comparison='greater'):
    d = _fbf[parameter].copy()
    try:
        if time_period == 'prestim':
            d = d[d.index <= d[(_fbf['Laser1_state'] + _fbf['Laser2_state']) > 0].index[0]]#.dropna()
        elif time_period == 'stim':
            d = d[(d.index >= d[(_fbf['Laser1_state'] + _fbf['Laser2_state']) > 0].index[0]) & (d.index <= d[(_fbf['Laser1_state'] + _fbf['Laser2_state']) > 0].index[-1])]#.dropna()
        if time_period == 'poststim':
            try:
                d = d[d.index > d[(_fbf['Laser1_state'] + _fbf['Laser2_state']) > 0].index[-1]]#.dropna()
            except:
                print "Error limiting index to time period: "
        if time_period == 'red':
            d = d[(d.index >= d[(_fbf['Laser2_state']) > 0].index[0]) & (d.index <= d[(_fbf['Laser2_state']) > 0].index[-1])]
        if time_period == 'between':
            d = d[(d.index >= d[(_fbf['Laser1_state']) > 0].index[-1]) & (d.index <= d[(_fbf['Laser2_state']) > 0].index[0])]
        if time_period == 'IR':
            d = d[(d.index >= d[(_fbf['Laser1_state']) > 0].index[0]) & (d.index <= d[(_fbf['Laser1_state']) > 0].index[-1])]
        add_something=False
    except:
        add_something=False
        pass
    
    if parameter == 'Ab_bending':
        targeted = get_bouts(_fbf, 'target_distance_TTM', 1.0, time_period, comparison='Less')
        numvisits = targeted['bout_duration'].count()
        period_length = targeted['bout_duration'].sum()
    else:
        period_length = (d.index[-1] - d.index[0]).total_seconds()
        numvisits = 1
    if comparison=='greater':
        if d.ix[d.index[0]] > threshold:
            add_something = True
    else:
        print d[0]
        try:
            if d.ix[d.index[0]] < threshold:
                add_something = True
        except:
            return
    d = d.fillna(method='pad', limit=3).dropna()
    if comparison=='greater':
        d[d<=threshold] = 0
        d[d>threshold] = 1
    else:
        d[d<=threshold] = -1e6
        d[d>threshold] = 1
        d[d == -1e6] = 0
        
    t = d - d.shift()
    ts = t[abs(t) >0]
    if comparison=='greater':
        ons = ts.index[ts == 1]
        offs = ts.index[ts == -1]
    else:
        ons = ts.index[ts == -1]
        offs = ts.index[ts == 1]
    lons = list(ons)
    loffs = list(offs)
    ontimes = []
    offtimes = []    
    for x in lons:
        ontimes.append((x - pd.to_datetime(0)).total_seconds() )
    for x in loffs:
        offtimes.append((x - pd.to_datetime(0)).total_seconds() )
    
    if add_something == True:
        ontimes.insert(0,0)
    
    if len(ontimes) > len(offtimes):
        offtimes.append((t.index[-1]- pd.to_datetime(0)).total_seconds())
    if len(offtimes) > len(ontimes):
        ontimes.insert(0,(t.index[0]- pd.to_datetime(0)).total_seconds())
    
    
    try:
        df = pd.DataFrame({'ons':ontimes, 'offs':offtimes})
        if len(df) == 0:
            df = pd.DataFrame({'ons':[0], 'offs':[0]})
            df['bout_duration'] = 0
            
        else:
            df['bout_duration'] = df['offs'] - df['ons']
        df['time_period'] = time_period
        df['period_length'] = period_length
        df['visits'] = float(numvisits)
            
    except:
        print len(ontimes), '\n\n\n', len(offtimes)
        print ontimes, '\n\n\n', offtimes
        df = pd.DataFrame({'ons':ons, 'offs':offs, 'bout_duration':np.nan, 'time_period':time_period, 'period_length':period_length})
    try:
        if parameter == 'Ab_bending':
            pass#df.loc[df['bout_duration'] <=0.05, 'bout_duration'] = 0
        #if (parameter == 'target_distance_TTM') & (threshold == 2.0):
        #df = df[df['bout_duration'] >= 0.125] #np.timedelta64(500,'ms')]
        #df.loc[df['bout_duration'] <=0.125, 'bout_duration'] = 0
        #df = df.loc[df['bout_duration'] <=0.1, 'bout_duration'] = 0
    except:
        df = pd.DataFrame({'ons':ontimes, 'offs':offtimes, 'bout_duration':0, 'time_period':time_period, 'period_length':period_length})
    return df

selected_INDEX= [0,1,2,6,10,11,12,14,18,21,24,28,32,36,37,38,39,40,41,43,47,49,53,55,56,57,58,62,63,64,65]

COMPARISON = pd.DataFrame({'comp':['T2','T3','G0','G0','T1','T1','T2','G11','G11','T2','G15',
              'G15','G65','T1','T1','T2','T3','T2','T1','G0','T1','G11',
              'G15','T2','T1','T1','G65','G65','T3','T2','T1']}, index=selected_INDEX)

def get_relevant(data):
    """
    if '.csv' in data:
        pvs = pd.read_csv(data, header=0, sep=',')
        pvs.columns = ['stat','old_index','group1','group2','p-value','time_period']
        pvs.index = pvs.old_index
    else: 
        pvs = data
        pvs.columns = ['stat','old_index','group1','group2','p-value','time_period']
    """
    bar = data.merge(COMPARISON, how='inner', left_index=True, right_index=True)
    #bar = bar[['stat','group1','group2','p-value','time_period','comp']]
    bar = bar.reset_index()
    return bar

colourlist = ['#0033CC','#33CC33','#FFAA00', '#CC3300', '#AAAAAA','#0032FF','r','c','m','y', '#000000', '#333333']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameter', type=str, required=True,
                            help='parameter for which to measure bouts')
    parser.add_argument('--threshold', type=float, required=True,
                            help='threshold to define bouts')
    parser.add_argument('--greater_than', type=str, required=False, default="greater",
                            help='change to "less" to select bouts of low value')
    parser.add_argument('--time', type=bool, required=False, default=False,
                            help='make true to plot data binned per minute')
    
    args = parser.parse_args()
    greater_than = args.greater_than
    if greater_than == 'greater':
        fillValue = 0
    else:
        fillValue = 1e9
    parameter = args.parameter
    threshold = args.threshold
    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'  
        
    colourlist = ['#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
    time_periods = ['poststim'] 

    #time_periods = ['prestim','stim','poststim']
    #time_periods = ['prestim','red','between','IR','poststim']
    #time_periods = ['prestim','between','poststim']
    #dataset = pd.DataFrame({'FlyID':[],'GROUP':[],'onset_time':[],'bout_duration':[], 'time_period':[]})
    dataset = pd.DataFrame()
    for directory in glob.glob(DATADIR  + '*zoom*'):
        #print 'processing: ', directory.split('/')[-1]
        try:
            fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        except:
            continue
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        if GROUP != FLY_ID.split('_')[0]:
            1/0
        if 'target_distance_TTM' in fbf.columns:
            fbf['target_distance_TTM'] = fbf['target_distance_TTM'].fillna(method='pad',limit=3).fillna(fbf['dtarget']).fillna(fbf['dtarget'].max())
        if 'group' in fbf.columns:
            GROUP = fbf['group'][0]
        fbf.loc[fbf['Length'] <= 245.0,'Length'] = np.nan
        
        if parameter == 'Ab_bending':
            fullLength = np.around(fbf.Length).mode()[0]
            if (parameter + '_smoothed') not in fbf.columns:
                fbf['Length'] = fbf['Length'].fillna(method='pad', limit=3).fillna(fullLength)
                fbf = smooth(fbf, 'Length')
                fbf.to_pickle(directory + '/frame_by_frame_synced.pickle')
            fbf.loc[(fbf.Length_smoothed < 0.95*fullLength) & (fbf.target_distance_TTM < 1.0), 'Ab_bending'] = 1
            
            fbf['Ab_bending'].fillna(method='pad', limit=1, inplace=True)
            fbf['Ab_bending'].fillna(0, inplace=True)
            fbf['Ab_bending'] = fbf['Ab_bending']*fbf['Ab_bending'].shift()*fbf['Ab_bending'].shift(-1).fillna(0)
                        
        for p in time_periods:
            print "defining bouts: ", directory.split('/')[-1]
            if (parameter + '_smoothed') not in fbf.columns:
                fbf[parameter] = fbf[parameter].fillna(method='pad', limit=5).fillna(fillValue)
                fbf = smooth(fbf, parameter)
                fbf.to_pickle(directory + '/frame_by_frame_synced.pickle')
            df = get_bouts(fbf, (parameter + '_smoothed'), threshold, p, greater_than)
            try:
                tempdf = pd.DataFrame({'FlyID':FLY_ID,'GROUP':GROUP,'onset_time':df['ons'],'offset_time':df['offs'],'bout_duration':df['bout_duration'], 'time_period':df['time_period'], 'period_length':df['period_length']})
                g1, g2, g3 = FLY_ID.split('_')[0].split('-',2)
                tempdf['g1'] = g1
                tempdf['g2'] = g2
                tempdf['g3'] = g3
                
                dataset = pd.concat([dataset, tempdf], axis=0)
                print "\t finished: ", directory.split('/')[-1]
            except:
                print FLY_ID, '\n',GROUP, '\n',df

        #except:
        #    print "failed to acquire data: ", directory.split('/')[-1]
    
    dataset['normalized_duration'] = 60.0*dataset['bout_duration'] / dataset['period_length']
    dataset.to_pickle(DATADIR + args.parameter +'_'+str(args.threshold)+ '_bouts.pickle')
    
    grouped = dataset.groupby(['time_period','GROUP','FlyID'])
    
    pvalues = pd.DataFrame()
    for G in sorted(list(set(dataset.g3))):
        d = dataset[dataset['g3'] == G]
        groups = sorted(list(set(d['GROUP'])))
        pairs = generate_all_pairs(groups)
        p_values = do_stats(pairs, d, 'GROUP','normalized_duration','poststim')
        pvalues = pd.concat([pvalues, p_values])
    print pvalues
    pvalues.to_csv(DATADIR + args.parameter +'_'+str(args.threshold)+ '_p_values_unweighted.csv')
    '''
    #GROUPING BY FLY:
    for G in sorted(list(set(dataset.g3))):
        d = dataset[dataset['g3'] == G]
        means = d.groupby('FlyID').mean().reset_index()
        counts = d.groupby('FlyID').count().reset_index()
        for row in means.index:
            means.loc[row, 'GROUP'] = means.loc[row, 'FlyID'].split('_')[0]

        counts['GROUP'] = means['GROUP']
        groups = sorted(list(set(d['GROUP'])))
        pairs = generate_all_pairs(groups)
        p_values = do_stats(pairs, counts, 'GROUP','normalized_duration','poststim')
        print p_values

    
    p_values_total_time = pd.DataFrame()
    for period, data in grouped.sum().groupby(level=[0]):
        data.reset_index(inplace=True)
        p_values_temp = do_stats(pairs, data, 'GROUP', 'normalized_duration', period)
        p_values_total_time = pd.concat([p_values_total_time, p_values_temp])

    if args.time:
        dataset['minute'] = np.floor(dataset['onset_time']/60.0)
        _grouped = dataset.groupby(['GROUP','FlyID','minute'])
        timeplot(_grouped.mean(), 'bout_duration', 'Mean bout duration')
        timeplot(_grouped.count(), 'bout_duration', 'Mean number of bouts')
        timeplot(_grouped.max(), 'bout_duration', 'Mean max bout duration')
        timeplot(_grouped.sum()/60.0, 'bout_duration', 'Index')    
    

    grouped = dataset.groupby(['time_period','GROUP','FlyID'])
    barplot(grouped.sum(), 'normalized_duration', 'Index')
    
    NZ_dataset = dataset[dataset['bout_duration'] >0]
    NZ_grouped = NZ_dataset.groupby(['time_period','GROUP','FlyID'])
    
    barplot(NZ_grouped.count(), 'normalized_duration', 'Mean number of bouts per minute')

    p_values_counts = pd.DataFrame()
    for period, data in NZ_grouped.count().groupby(level=[0]):
        data.reset_index(inplace=True)
        p_values_temp = do_stats(pairs, data, 'GROUP', 'normalized_duration', period)
        p_values_counts = pd.concat([p_values_counts, p_values_temp])
    
    p_values_duration = pd.DataFrame()
    for period, data in grouped.mean().groupby(level=[0]):
        data.reset_index(inplace=True)
        p_values_temp = do_stats(pairs, data, 'GROUP', 'bout_duration', period)
        p_values_duration = pd.concat([p_values_duration, p_values_temp])
    p_values_max = pd.DataFrame()
    for period, data in grouped.max().groupby(level=[0]):
        data.reset_index(inplace=True)
        p_values_temp = do_stats(pairs, data, 'GROUP', 'bout_duration', period)
        p_values_max = pd.concat([p_values_max, p_values_temp])

    barplot(grouped.mean(), 'bout_duration', 'Mean duration')
    barplot(grouped.max(), 'bout_duration','Max. duration')
    '''
    grouped = dataset.groupby(['g1','g3','FlyID'])
    lineplot(grouped.mean(), 'bout_duration', 'Mean duration')
    lineplot(grouped.max(), 'bout_duration','Max. duration')
    lineplot(grouped.sum(), 'normalized_duration', 'Index')
    
    NZ_grouped = NZ_dataset.groupby(['g1','g3','FlyID'])
    lineplot(NZ_grouped.count(), 'normalized_duration','Mean number of bouts per minute')
    
    
    pvals = pd.concat([get_relevant(p_values_duration), get_relevant(p_values_counts), get_relevant(p_values_total_time), get_relevant(p_values_max)], keys=['boutLength','count','totalTime','max'])
    #pvals = get_relevant(pvals.reset_index())
    pvals.to_csv(DATADIR + args.parameter +'_'+str(args.threshold)+ '_p_values.csv')
    print "P-VALUES: \n", pvals


    
    
        
    
  
        
