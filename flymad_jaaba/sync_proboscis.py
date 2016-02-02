    
    
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections    
import argparse
import glob
import os
from scipy import stats as st
from scipy.stats import kruskal
import flymad_jaaba.flymad_jaaba_v6 as flymad
    
    
  
    
    
    
def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id



prestim = np.timedelta64(5,'s')
poststim = np.timedelta64(10,'s')

def sync_by_stims(fbf, p, FlyID, GROUP):
    df = pd.merge(p, fbf, right_on='Frame_number', left_index=True)
    d = df['Laser2_state'].copy()
    t = d - d.shift()
    ts = t[abs(t) >0]
    ons = ts.index[ts == 1]
    offs = ts.index[ts == -1]
    synced_df = pd.DataFrame({'Frame_number':[],'Laser2_state':[],'length':[],'stim_number':[],'synced_seconds':[]})
    for x in range(len(ons)):
        slyce = df[ons[x]-prestim:ons[x]+poststim].copy()
        slyce['synced_time'] -= slyce.ix[0]['synced_time'] + prestim
        slyce.index = slyce['synced_time']
        slyce.index = pd.to_datetime(slyce.index)
        #slyce = slyce.resample('250ms', how='mean')
        times = []
        for y in slyce.index:
            times.append((y - pd.to_datetime(0)).total_seconds())
        slyce['synced_seconds'] = times
        slyce['stim_number'] = x+1
        l = slyce['length'].copy()
        l[l>0] = 1.0
        l = l*(l.shift() + l.shift(-1))
        l[l>0] = 1.0
        slyce['length'] *=l
        synced_df = pd.concat([synced_df, slyce[['Frame_number','Laser2_state','length','stim_number','synced_seconds']]])
    synced_df['group'] = GROUP
    synced_df['FlyID'] = FlyID
    synced_df['Proboscis extension'] = 0
    synced_df.loc[synced_df['length'] >0,'Proboscis extension'] = 1
    synced_df['rounded_t'] = np.around(synced_df['synced_seconds'], decimals=1)
    return synced_df

def plot_traces(data, param, group, filename):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if 'length' in param:
        max_scale = max([12, 1.1*((data[param].mean() + data[param].sem()).max())])
    else:
        max_scale = min([1,1.3*((data[param].mean() + data[param].sem()).max())])
    laser_2 = collections.BrokenBarHCollection.span_where(data[param].mean().index.values, ymin=0, ymax=max_scale, where=data.mean()['Laser2_state'] > 0.1, facecolor='#FFCCCC', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #red FFB2B2
    ax.add_collection(laser_2)
    
    q = plt.fill_between(data[param].mean().index.values, 
                        data[param].mean() + data[param].sem(), 
                        data[param].mean() - data[param].sem(), 
                        alpha=0.1, 
                        color='#202090',
                        zorder=90)  
    p = plt.plot(data[param].mean().index.values, data[param].mean(), 
                        color='#202090',linewidth=3, zorder=100)

    ax.set_title(group + ' N=' + str(max(data[param].count())), fontsize=18)
    ax.set_ylabel('Mean ' + param  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
    ax.set_ylim(0,max_scale)
    ax.set_xlabel('Time (s)', fontsize=16)      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')  
    
    plt.savefig(filename+'.png')
    plt.savefig(filename+'.svg')
    plt.close('all')
    return  
    
def iterate_groups(full_df):    
    groups = list(set(full_df['group']))
    
    for group in groups:
        df = full_df[full_df['group'] == group]
        g = df.groupby(['FlyID','rounded_t']).mean().reset_index()
        h = g.groupby('rounded_t')
        plot_traces(h, 'length', group, (DATADIR+'PLOTS/'+group+'_length'))
        plot_traces(h,'Proboscis extension', group, (DATADIR+'PLOTS/'+group+'_PE'))
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    #parser.add_argument('--threshold', type=float, required=True,
    #                        help='threshold to define bouts')
    #parser.add_argument('--greater_than', type=str, required=False, default="greater",
    #                        help='change to "less" to select bouts of low value')
    
    args = parser.parse_args()

    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'    
    
    dataset = pd.DataFrame({'Frame_number':[],'Laser2_state':[],'length':[],'stim_number':[],'synced_seconds':[]})
    for directory in glob.glob(DATADIR  + '*zoom*'):
        print 'processing: ', directory.split('/')[-1]
        try:
            FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
            _fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
            _p = pd.read_pickle(directory + '/proboscis_data.pickle')
            tempdf = sync_by_stims(_fbf, _p, FLY_ID, GROUP)
            dataset = pd.concat([dataset, tempdf], axis=0)
        except:
            print "ERROR processing file."
            pass
    dataset.to_pickle(DATADIR+ 'proboscis_data_synced.pickle')
    dataset.to_csv(DATADIR+ 'proboscis_data_synced.csv',sep=',')    
    iterate_groups(dataset)    
    if not (os.path.exists(DATADIR + 'STATS') ==True):
        os.makedirs(DATADIR + 'STATS')
    fname_prefix = DATADIR + '/STATS/p_values_'
    groups = list(set(dataset['group']))
    test_groups = ['M','F','v16','v17']
    
    for x in test_groups:
        df = dataset[dataset['group'].str.contains(x)]
        groups = list(set(df['group']))
        pp = flymad.view_pairwise_stats(df, groups, 'v16', (fname_prefix + 'PE_v16_' + x),
                                    stat_colname='Proboscis extension',
                                    layout_title='Kruskall-Wallis H-test: Proboscis extension',
                                    num_bins=15)
        pp = flymad.view_pairwise_stats(df, groups, 'v16', (fname_prefix + 'length_v16_'+x),
                                    stat_colname='length',
                                    layout_title='Kruskall-Wallis H-test: Length',
                                    num_bins=15)  
        pp = flymad.view_pairwise_stats(df, groups, 'v17', (fname_prefix + 'PE_v17_' + x),
                                    stat_colname='Proboscis extension',
                                    layout_title='Kruskall-Wallis H-test: Proboscis extension',
                                    num_bins=15)
        pp = flymad.view_pairwise_stats(df, groups, 'v17', (fname_prefix + 'length_v17_'+x),
                                    stat_colname='length',
                                    layout_title='Kruskall-Wallis H-test: Length',
                                    num_bins=15)      

    """
    for group in groups:
        pp = flymad.view_pairwise_stats(dataset, [group, 'v-16-M-stop'], (fname_prefix + group + '_PE'),
                                        stat_colname='Proboscis extension',
                                        layout_title='Kruskall-Wallis H-test: Proboscis extension, v-16-M-stop vs ' + group,
                                        num_bins=15)
        pp = flymad.view_pairwise_stats(dataset, [group, 'v-16-M-stop'], (fname_prefix + group + '_length'),
                                        stat_colname='length',
                                        layout_title='Kruskall-Wallis H-test: Length, v-16-M-stop vs ' + group,
                                        num_bins=15)
        pp = flymad.view_pairwise_stats(dataset, [group, 'v-17-M-stop'], (fname_prefix + group + '_PE'),
                                        stat_colname='Proboscis extension',
                                        layout_title='Kruskall-Wallis H-test: Proboscis extension, v-17-M-stop vs ' + group,
                                        num_bins=15)
        pp = flymad.view_pairwise_stats(dataset, [group, 'v-17-M-stop'], (fname_prefix + group + '_length'),
                                        stat_colname='length',
                                        layout_title='Kruskall-Wallis H-test: Length, v-17-M-stop vs ' + group,
                                        num_bins=15)
        
    """
