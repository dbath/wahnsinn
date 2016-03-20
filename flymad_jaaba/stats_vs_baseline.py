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
<<<<<<< HEAD
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

def plot_stats( data, fig_prefix, **kwargs):
    """ data = output from gather_data().   
        names = list of groups (ex. ['foo','bar','baz'])
        fig_prefix = full path and filename (without extension) of plot name.
        **kwargs = 
    """

    graph_data = []
    layout=None
    pvalue_results = {}
    for ROI in data.columns[1:]:
        pairwise_data = get_pairwise( data, ROI, **kwargs)
        if pairwise_data is not None:
            graph_data.append( pairwise_data['data'] )
            layout=pairwise_data['layout']
            pvalue_results[ROI] = pairwise_data['df']
        else:
            print "pairwise_data is none: ", ROI

    if len( graph_data )==0:
        print 'length is 0'
        pairwise_data = get_pairwise( data, stat_colname=ROI, **kwargs)
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

def get_pairwise(data, stat_colname, **kwargs):
    layout_title = kwargs.pop('layout_title',None)
    #human_label_dict = kwargs.pop('human_label_dict',None)
    p_values = calc_p_values(data, stat_colname, **kwargs)
    
def get_pairwise(data,**kwargs):
    layout_title = kwargs.pop('layout_title',None)
    #human_label_dict = kwargs.pop('human_label_dict',None)
    p_values = calc_p_values(data, **kwargs)
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
    parser.add_argument('--binsize', type=str, required=True,
                            help='integer and unit, such as "5s" or "4Min" or "500ms"')
    
    
       
    args = parser.parse_args()

    DATADIR = args.rawdatadir
    if not DATADIR[-1] == '/' :
        DATADIR = DATADIR + '/'



    binsize = (args.binsize)
    print "BINSIZE: ", binsize
    colourlist = ['#202090','#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
    #colourlist = ['#2020CC','#20CC20','#FFCC20','#CC2000','#202020']
    #colourlist = ['#CC2000','#2020CC','#20CC20','#FFCC20','#CC2000','#202020']
    #filename = '/tier2/dickson/bathd/FlyMAD/DATADIR_tracking/140927/wing_angles_nano.csv'
    kinetics = DataFrame({'ROI':[], 'ID':[], 'OFFTIME':[]})
    for pickle in glob.glob(DATADIR + 'rawdata/*.pickle'):
        print pickle
        fname_prefix = pickle.rsplit('/',2)[0] + '/'+ pickle.split('/')[-1].split('.csv')[0] + '_p-values_'
        _df = pd.read_pickle(pickle)
        if 'time' in _df.columns:
            _df['Time'] = _df['time']    #silly inconsistent data formats.
        _df['Time'] = _df['Time'].astype(np.float64)    
        pp = plot_stats(_df, fname_prefix,
                                       layout_title=('Kruskal-Wallis H-test: baseline'),
                                       num_bins=120,
                                       )
        df = DataFrame()
        offtimes = []
        for x in pp:
            _df = DataFrame(pp[x])
            try:
                offtime = ( _df[(_df['P'] > 0.05/120.0) & (_df['bin_start_time'] > 60.0)]['bin_stop_time'].index[0]  )
            except:
                offtime = _df.loc[_df.bin_stop_time.argmax(),'bin_stop_time']
            print offtime
            kinetics = kinetics.append({'ROI':pickle.split('/')[-1].split('.pickle')[0], 'ID':x, 'OFFTIME':offtime}, ignore_index=True)
            df = pd.concat([df, _df], axis=0)
        
        #print df[0:5]
        df.to_pickle(fname_prefix + '.pickle')
    kinetics.to_csv(DATADIR + 'offtimes.csv', sep=',')

    for pickle in glob.glob(DATADIR + '*.pickle'):
        fname_prefix = pickle.split('.pickle')[0] + '_p-values_'
        _df = pd.read_pickle(pickle)
        pp = plot_stats(_df, fname_prefix,
                                       layout_title=('Kruskal-Wallis H-test: baseline'),
                                       num_bins=602,
                                       )




