import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats as st


colourlist = ['#333333','#0033CC',  '#AAAAAA','#0032FF','r','c','m','y', '#000000']

def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def get_length_dist(df, _bins=np.arange(0,1.6,0.05)):

    post_dist = plt.hist(df[df['synced_time']>0.0].maxWingAngle, bins=_bins)[0]
    postcount = sum(post_dist)
    pre_dist = plt.hist(df[df['synced_time']<0.0].maxWingAngle, bins=_bins)[0]
    precount = sum(pre_dist)  
    return (pre_dist / precount), (post_dist / postcount)
    

def group_data(df):
    df.index = df['Angle']
    grouped = df.groupby(['group', df.index])
    means = grouped.mean()
    ns = grouped.count()
    sems = grouped.aggregate(lambda x: st.sem(x, axis=None))
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
            x_values.append(means.ix[x,w]['Angle'])
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
    ax.set_ylim(0.85*(np.amin(y_range)),1.15*(np.amax(y_range)))
    ax.set_xlabel('Wing Angle (rad)', fontsize=22) 
    ax.set_ylabel(measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=22)
    l = plt.legend()
    l.set_zorder(1000)
    fig.savefig(DATADIR +'wingAngleDist_' + HANDLE + plotID + '_mean.svg', bbox_inches='tight')
    fig.savefig(DATADIR +'wingAngleDist_' + HANDLE + plotID + '_mean.pdf', bbox_inches='tight')
    fig.savefig(DATADIR +'wingAngleDist_' + HANDLE + plotID + '_mean.png', bbox_inches='tight')







pre_df = pd.DataFrame({'Angle':[],'Frequency':[],'FlyID':[],'GROUP':[]})
post_df = pd.DataFrame({'Angle':[],'Frequency':[],'FlyID':[],'GROUP':[]})


DATADIR = '/media/DBATH_7/150707/'
HANDLE = 'SS01538'

for directory in glob.glob(DATADIR + '*' + HANDLE + '*' + '*zoom*'):
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
    TRACKING_DATA = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
    pre_data, post_data = get_length_dist(TRACKING_DATA)
    TEMPpre_df = pd.DataFrame({'Angle':np.arange(0,1.6,0.05)[:-1], 'Frequency':pre_data, 'FlyID':FLY_ID, 'group':GROUP})
    TEMPpost_df = pd.DataFrame({'Angle':np.arange(0,1.6,0.05)[:-1], 'Frequency':post_data, 'FlyID':FLY_ID, 'group':GROUP})
    pre_df = pd.concat([pre_df, TEMPpre_df], axis=0)
    post_df = pd.concat([post_df, TEMPpost_df], axis=0)

plt.close('all')
    
means, sems, ns = group_data(pre_df)
plot_data(means, sems, ns, 'Frequency', 'baseline')
means, sems, ns = group_data(post_df)
plot_data(means, sems, ns, 'Frequency', 'postStimulus')

    
