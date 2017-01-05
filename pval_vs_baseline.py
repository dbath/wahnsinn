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
from math import isnan
import flymad_jaaba.flymad_jaaba_v6
import flymad.fake_plotly as fake_plotly
import pprint
import traceback

NANOSECONDS_PER_SECOND = 1000000000



def calc_p_values(_data,
                  stat_colname=None,
                  num_bins=50, bin_how='mean',
                  ):

    if stat_colname is None:
        raise ValueError("you must explicitly set stat_colname (try 'maxWingAngle')")
    
    _data.index = _data.Time  #LAZY DANNO. DROP TIMESTAMPS FOR BINNING.
    _data = _data.sort('Time')
    _data['synced_ns'] = _data.index
    df_baseline = _data[_data['Time'] < 10.0]
    align_start = _data.Time.min()
    dalign = int(_data.Time.max()) - int(align_start)
    dalign = _data.Time.max() - align_start
    p_values = DataFrame()

    if bin_how=='mean':
        bin_func = np.mean
    elif bin_how=='median':
        bin_func = np.median

    bins = np.linspace(0,dalign,num_bins+1) + align_start
    binned_data = pd.cut(_data.index, bins, labels= bins[:-1])

    baseline = df_baseline[stat_colname].values
    bin_number = 0
    for x in binned_data.levels:
        #test_df = data.loc[(data.index > binned_data.levels[x]) & (data.index <= binned_data.levels[x+1]), stat_colname].values
        test_df = _data.loc[binned_data == x, stat_colname]
        bin_start_time = x
        bin_stop_time = _data.loc[binned_data == x, 'Time'].max()
        test = np.array(test_df)
        
        try:
            hval, pval = kruskal(baseline, test)
        except ValueError as err:
            pval = 1.0

        dftemp = DataFrame({'Bin_number': bin_number,
                            'P': pval,
                            'bin_start_time':bin_start_time,
                            'bin_stop_time':bin_stop_time,
                            'name1':'baseline',
                            'name2':stat_colname,
                            'test1_n':len(baseline),
                            'test2_n':len(test),
                            }, index=[x])
        p_values = pd.concat([p_values, dftemp])
        bin_number +=1



    return p_values

def plot_stats( groupedData, fig_prefix, cutoff='baseline',  **kwargs):
    """ data = output from flymad_jaaba_v6.py (rawdata_**s.pickle), with synced_time column representing seconds, grouped by 'group'.   
        names = list of groups (ex. ['foo','bar','baz'])
        fig_prefix = full path and filename (without extension) of plot name.
        **kwargs = 
    """

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    
    for GROUP, data in groupedData:
        colour = colourlist[groupedData.groups.keys().index(GROUP)]
        pvalue_results = {}
        if cutoff == 'baseline':
            ax.set_title('Kruskal Wallis: '+parameter+' vs baseline:', fontsize=12)
            baseline = data[data.synced_time <=0][parameter].values    
            for time, _data in data[(data.synced_time > 0) & (data.synced_time <= 360)].groupby('synced_time'):
                pvalue_results[time*args.binsize] = st.kruskal(baseline, _data[parameter])[1]
        elif cutoff == 'zero':
            ax.set_title('Kruskal Wallis: '+parameter+' vs zero:', fontsize=12)
            for time, _data in data[(data.synced_time > 0) & (data.synced_time <= 360)].groupby('synced_time'):
                pvalue_results[time*args.binsize] = st.ttest_1samp(_data[parameter], 0)[1]

        pvalue_results = {k: pvalue_results[k] for k in pvalue_results if not isnan(pvalue_results[k])} 

        ax.scatter(pvalue_results.keys(), -np.log10(pvalue_results.values()), label=GROUP, color=colour, linewidth=0)
    
    if len(pvalue_results)>=1:
        n_comparisons = len(pvalue_results)
        ax.axhline( -np.log10(0.05/n_comparisons), color='k', lw=0.5, linestyle='--' )

    ax.set_xlim(0,360)
    ax.set_ylim(0,8)#1.1*max(-np.log10(pvalue_results.values())))
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('-Log10(P)', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    l = plt.legend()
    l.set_zorder(1000)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    for ext in ['.png','.svg']:
        fig_fname = fig_prefix + '_'+parameter + ext
        fig.savefig(fig_fname, bbox='tight')
        print 'saved',fig_fname

    return pvalue_results

def get_pairwise(data, stat_colname, **kwargs):
    layout_title = kwargs.pop('layout_title',None)
    #human_label_dict = kwargs.pop('human_label_dict',None)
    p_values = calc_p_values(data, stat_colname, **kwargs)
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
        xs.append( starts[i]  )
        ys.append( logs[i] )
        texts.append( 'p=%.3g, n=%d,%d t=%s to %s'%(
            pvals[i], n1[i], n2[i], starts[i], stops[i] ) )

        xs.append( stops[i]  )
        ys.append( logs[i] )
        texts.append( '')

    this_dict = {
        'name':stat_colname,
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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rawdatadir', type=str, required=True,
                            help='directory of fmf and bag files')  
    #parser.add_argument('--outputdir', type=str, required=True,
    #                        help='directory to store analysis')
    #parser.add_argument('--bagdir', type=str, required=True,
    #                        help='directory of bag files')
    parser.add_argument('--binsize', type=int, required=False, default=5,
                            help='integer representing number of seconds per bin')
    parser.add_argument('--parameters', type=str, required=False, default='maxWingAngle',
                            help='enter a list of flymad attributes. default is maxWingAngle')
    parser.add_argument('--downsample', type=bool, required=False, default=False,
                            help='make true to downsample to 1Hz resolution')
    
    
       
    args = parser.parse_args()
    parameters = [str(item) for item in args.parameters.split(',')]
    DATADIR = args.rawdatadir
    if not DATADIR[-1] == '/' :
        DATADIR = DATADIR + '/'



    binsize = (args.binsize)
    colourlist = ['#AAAAAA','#009020','#FFAA00', '#CC3300', '#AAAAAA','#0032FF','r','c','m','y', '#000000', '#333333']
    #colourlist = ['#202090','#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
    #colourlist = ['#2020CC','#20CC20','#FFCC20','#CC2000','#202020']
    #colourlist = ['#CC2000','#2020CC','#20CC20','#FFCC20','#CC2000','#202020']
    #filename = '/tier2/dickson/bathd/FlyMAD/DATADIR_tracking/140927/wing_angles_nano.csv'
    
    
    for pickle in glob.glob(DATADIR + 'JAR/*rawdata*.pickle'):
        print pickle
        #fname_prefix = pickle.rsplit('/',2)[0] + '/'+ pickle.split('/')[-1].split('.pickle')[0] + '_p-values_'
        _df = pd.read_pickle(pickle)
        _df['synced_time'] = ((_df.index.astype(int)/1e9) / args.binsize).astype(int)
        _df = _df.reset_index()
    
    for parameter in parameters:
   
        pp = plot_stats(_df.groupby('group'), DATADIR+'p-values_vs_baseline_', 'baseline'
                                       )
        pp = plot_stats(_df.groupby('group'), DATADIR+'p-values_vs_zero_', 'zero'
                                       )
        plt.close('all')





