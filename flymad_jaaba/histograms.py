import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats as st
import argparse


colourlist = ['#333333','#0033CC',  '#AAAAAA','#0032FF','r','c','m','y', '#000000']

def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def get_dist(df, _bins=np.arange(180,400,5)):

    post_dist = plt.hist(df[df['synced_time']>0.0][parameter], bins=_bins)[0]
    postcount = sum(post_dist)
    pre_dist = plt.hist(df[df['synced_time']<0.0][parameter], bins=_bins)[0]
    precount = sum(pre_dist)  
    return (pre_dist / precount), (post_dist / postcount)
    

def group_data(df):
    df.index = df[parameter]
    grouped = df.groupby(['group', df.index])
    means = grouped.mean()
    ns = grouped.count()
    sems = grouped.sem()
    #sems = grouped.aggregate(lambda x: np.std(x, ddof=1)/np.sqrt(x.count()))  #HANDLES NAN DATA BETTER THAN GROUPED.SEM()
    #sems = grouped.aggregate(lambda x: st.sem(x, axis=None))
    return means, sems, ns


def plot_data(means, sems, ns, measurement, plotID):
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
        
        
        for w in means.ix[x].index:
            #print ns.ix[x]['FlyID'][w]
            x_values.append(means.ix[x,w][parameter])
            y_values.append(means.ix[x,w][measurement])
            psems.append(sems.ix[x,w][measurement])
            nsems.append(-1.0*sems.ix[x,w][measurement])
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
    if args.log_bins == True:
        ax.set_xscale("symlog")
    ax.set_ylim(0.85*(np.amin(y_range)),1.3*(np.amax(y_range)))
    ax.set_xlabel(parameter, fontsize=22) 
    ax.set_ylabel(measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=22)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    l = plt.legend()
    l.set_zorder(1000)
    fig.savefig(DATADIR + plotID + '_' + parameter + '_mean.svg', bbox_inches='tight')
    fig.savefig(DATADIR + plotID + '_' + parameter + '_mean.pdf', bbox_inches='tight')
    fig.savefig(DATADIR + plotID + '_' + parameter + '_mean.png', bbox_inches='tight')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameter', type=str, required=True,
                            help='parameter for which to measure bouts')
    parser.add_argument('--min', type=float, required=False, default=180,
                            help='minimum range of histogram')
    parser.add_argument('--max', type=float, required=False, default=400,
                            help='maximum range of histogram')
    parser.add_argument('--bins', type=float, required=False, default=50,
                            help='number of bins for histogram')
    parser.add_argument('--log_bins', type=bool, required=False, default=False,
                            help='make True to generate a log-transformed set of bins.')


    args = parser.parse_args()
    parameter = args.parameter
    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'  
    
    if args.log_bins == True:
        if args.min < 0:
            BINS = np.logspace(0, np.log10(args.max - args.min + 1.0), args.bins) - 1.0 + args.min
        elif args.min == 0:
            BINS = np.logspace(0, np.log10(args.max+1.0), args.bins) - 1.0
        else:
            BINS = np.logspace(np.log10(args.min), np.log10(args.max), args.bins)
    else:
        BINS = np.linspace(args.min, args.max, args.bins)
    
        
    colourlist = ['#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
    pre_df = pd.DataFrame({parameter:[],'Frequency':[],'FlyID':[],'GROUP':[]})
    post_df = pd.DataFrame({parameter:[],'Frequency':[],'FlyID':[],'GROUP':[]})

    FULL_post = pd.DataFrame({parameter:[],'group':[]})
    FULL_pre = pd.DataFrame({parameter:[],'group':[]})


    for directory in glob.glob(DATADIR + '*zoom*'):
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        TRACKING_DATA = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        if 'target_distance_TTM' in TRACKING_DATA.columns:
            TRACKING_DATA['target_distance_TTM'] = TRACKING_DATA['target_distance_TTM'].fillna(method='pad',limit=2).fillna(TRACKING_DATA['dtarget']/5.0)
            TRACKING_DATA['dtarget'] = TRACKING_DATA['target_distance_TTM']
        pre_data, post_data = get_dist(TRACKING_DATA, BINS)
        TEMPpre_df = pd.DataFrame({parameter:BINS[:-1], 'Frequency':pre_data, 'FlyID':FLY_ID, 'group':GROUP})
        TEMPpost_df = pd.DataFrame({parameter:BINS[:-1], 'Frequency':post_data, 'FlyID':FLY_ID, 'group':GROUP})
        TEMP_full_post = TRACKING_DATA[TRACKING_DATA['synced_time'] > 0][parameter]
        TEMP_full_pre = TRACKING_DATA[TRACKING_DATA['synced_time'] < 0][parameter]
        TEMP_full_post['group'] = GROUP
        TEMP_full_pre['group'] = GROUP
        pre_df = pd.concat([pre_df, TEMPpre_df], axis=0)
        post_df = pd.concat([post_df, TEMPpost_df], axis=0)
        FULL_post = pd.concat([FULL_post, TEMP_full_post], axis=0)
        FULL_pre = pd.concat([FULL_pre, TEMP_full_pre], axis=0)
        #print "added: ", directory.split('/')[-1], FULL_post.shape
    plt.close('all')

    #groups = set(FULL_post['group'])

    #pre_D, pre_P = st.ks_2samp(FULL_pre[FULL_pre['group'] == groups[0]], FULL_pre[FULL_pre['group'] == groups[1]])
    #post_D, post_P = st.ks_2samp(FULL_post[FULL_post['group'] == groups[0]], FULL_post[FULL_post['group'] == groups[1]])

    #print "PRE:  D=",pre_D," p=",pre_P,' POST: D=',post_D,' p=',post_P

    means, sems, ns = group_data(pre_df)
    plot_data(means, sems, ns, 'Frequency', 'baseline')
    means, sems, ns = group_data(post_df)
    plot_data(means, sems, ns, 'Frequency', 'postStimulus')



    
