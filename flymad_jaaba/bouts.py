from scipy.stats import ttest_ind
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse


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
    
def do_stats(pairs, dataset, group_col, stat_col):
    pvalue_results = {}
    for pair in pairs:
        name1, name2 = pair
        g1 = dataset[dataset[group_col] == name1]
        g2 = dataset[dataset[group_col] == name2]
        pvalue_results[pair] = ttest_ind(g1[stat_col], g2[stat_col])[1]
    return pvalue_results    

 
def barplot(grouped_df, statistic):
    set_of_groups = set(dataset['GROUP'])
    bar_width = 1.0/(len(set_of_groups))
    error_config = {'ecolor': '0.1'}
    means = grouped_df.groupby(level=[0]).mean()
    sems = grouped_df.groupby(level=[0]).sem()
    stds = grouped_df.groupby(level=[0]).std()
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    
    plt.bar(np.arange(0.1,(len(set_of_groups)+0.1),1), 
            means['bout_duration'],
            bar_width, 
            color= '#AAAAAA',
            yerr=sems['bout_duration'],
            error_kw=error_config,
            label=list(set_of_groups))
    
    ax.set_ylim(0,1.3*(means['bout_duration'].values.max()))
    if means['bout_duration'].values.min() >= 0:
        ax.set_ylim(0,1.1*((means['bout_duration'] + sems['bout_duration']).values.max()))
    else:
        ax.set_ylim(1.1*((means['bout_duration'] - sems['bout_duration']).values.min()),1.1*((means['bout_duration'] + sems['bout_duration']).values.max()))
    ax.set_ylabel(statistic + ' ' + args.parameter + ' ' + u"\u00B1" + ' SEM', fontsize=20)   # +/- sign is u"\u00B1"
    ax.set_xticks(np.arange(0.1+bar_width/2.0,(len(set_of_groups)+0.1+(bar_width/2.0)),1)) # (bar_width*(len(list_of_genotypes)/2)))
    ax.set_xticklabels(sorted(list(set_of_groups)))
    ax.tick_params(axis='y', labelsize=16 )
    ax.set_xlabel('Genotype', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    #plt.legend(prop={'size':10})           
    plt.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) + '_'+statistic.replace(' ','_')+'.svg', bbox_inches='tight')
    return

def timeplot(grouped_df, measurement, statistic):
        
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    group_number = 0
    for x in grouped_df.groupby(level=0):
        mean = pd.Series(x[1].groupby(level=[0,2]).mean()[measurement])
        
        sem  = pd.Series(x[1].groupby(level=[0,2]).sem()[measurement])
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
    ax.set_ylabel(statistic+' ' + measurement  + ' bout duration' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Minutes since stimulus onset', fontsize=16)    
    #ax.set_xlim((np.amin(),np.amax(x_range)))
    #ax.set_ylim(0.85*(np.amin(grouped_df.mean()[measurement]),1.15*(np.amax(grouped_df.mean()[measurement]))))
    l = plt.legend()
    l.set_zorder(1000)
    fig.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) +'_'+statistic+ '_'+ measurement +'_vs_time.svg', bbox_inches='tight')
    
def get_bouts(_fbf, parameter, threshold, comparison='greater'):
    d = _fbf[parameter].copy()
    d = d[d.index >= pd.to_datetime(0)]#.dropna()
    add_something=False
    if comparison=='greater':
        if d[d.index == pd.to_datetime(0)][0] > threshold:
            add_something = True
    else:
        if d[d.index == pd.to_datetime(0)][0] < threshold:
            add_something = True
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
        df['bout_duration'] = df['offs'] - df['ons']
    except:
        print len(ontimes), '\n\n\n', len(offtimes)
        print ontimes, '\n\n\n', offtimes
    try:
        df[df['bout_duration'] >= 0.5] #np.timedelta64(500,'ms')]
    except:
        df = pd.DataFrame({'ons':ontimes, 'offs':offtimes, 'bout_duration':np.nan})
    return df

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
    parameter = args.parameter
    threshold = args.threshold
    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'  
        
    colourlist = ['#333333','#0033CC', '#AAAAAA','#6699FF', '#202020','#0032FF','r','c','m','y', '#000000']  

    dataset = pd.DataFrame({'FlyID':[],'GROUP':[],'onset_time':[],'bout_duration':[]})
    for directory in glob.glob(DATADIR  + '*zoom*'):
        print 'processing: ', directory.split('/')[-1]
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        if 'target_distance_TTM' in fbf.columns:
            fbf['target_distance_TTM'] = fbf['target_distance_TTM'].fillna(method='pad',limit=2).fillna(fbf['dtarget']/5.0)
        
        fbf.loc[fbf['Length'] <= 275,'Length'] = np.nan
        df = get_bouts(fbf, parameter, threshold, greater_than)
        tempdf = pd.DataFrame({'FlyID':FLY_ID,'GROUP':GROUP,'onset_time':df['ons'],'bout_duration':df['bout_duration']})
        dataset = pd.concat([dataset, tempdf], axis=0)
    dataset.to_pickle(DATADIR + args.parameter +'_'+str(args.threshold)+ '_bouts.pickle')
    grouped = dataset.groupby(['GROUP','FlyID'])
    means = grouped.mean().groupby(level=[0]).mean()
    sems = grouped.mean().groupby(level=[0]).sem()
    stds = grouped.mean().groupby(level=[0]).std()
    
    pairs = generate_all_pairs(list(set(dataset['GROUP'])))
    p_values = do_stats(pairs, dataset, 'GROUP', 'bout_duration')
    print 'DURATION: \n', p_values
    
    counts = dataset.groupby(['GROUP','FlyID']).count().reset_index()
    p_values_counts = do_stats(pairs, counts, 'GROUP','bout_duration')
    
    print 'NUMBER OF BOUTS:\n', p_values_counts
    #pd.DataFrame(p_values, index=None).to_pickle('/nearline/dickson/bathd/FlyMAD_data_archive/P1_activation_and_adt2_silencing/pool/DATA_ANALYSIS/visits_p.pickle')
    
    barplot(grouped.mean(), 'Mean bout duration')
    barplot(grouped.count(), 'Mean number of bouts')
    barplot(grouped.max(), 'Max. bout duration')
    barplot(grouped.sum(), 'Total time')
    
    if args.time:
        dataset['minute'] = np.floor(dataset['onset_time']/60.0)
        _grouped = dataset.groupby(['GROUP','FlyID','minute'])
        timeplot(_grouped.mean(), 'bout_duration', 'Mean')
        timeplot(_grouped.count(), 'bout_duration', 'Number of')
        timeplot(_grouped.max(), 'bout_duration', 'Mean max')
        timeplot(_grouped.sum()/60.0, 'bout_duration', 'Index')
        
    
  
        
