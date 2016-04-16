from collections import Counter
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob


def binarize(_Series, threshold, greater_than=True):
    Series = _Series.copy()
    if greater_than:
        Series[Series <= threshold] = 0
        Series[Series > threshold] = 1
    else:
        Series[Series >= threshold] = np.nan
        Series[Series < threshold] = 1
    Series = Series.fillna(0)
    return np.array(Series)

def transition_matrix(Array):
    """pass a numpy array of dim (n,1) with 1 or 0 for state"""
    if len(set(Array)) == 1:
        return np.array([[0,0],[0,1]])
    else:
        pmat = np.zeros((len(set(Array)),(len(set(Array)))))
        for (x,y), c in Counter(zip(Array, Array[1:])).iteritems():
            pmat[x-1,y-1] = c
        for a in range(len(pmat)):
            pmat[a] = pmat[a] / float(pmat[a].sum())
        return pmat


def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

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
        ax.set_ylim(0.98*((meansALL[_column] - semsALL[_column]).values.min()),1.1*((meansALL[_column] + semsALL[_column]).values.max()))
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
    plt.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) + '_'+statistic.replace(' ','_')+'_transition_probabilties.svg', bbox_inches='tight')
    return


    
    
treatments = ['00','11','15','65']
genotypes = ['DB072','DB185','DB213']
colourlist = ['#0033CC','#33CC33','#FFAA00', '#CC3300', '#AAAAAA','#0032FF','r','c','m','y', '#000000', '#333333']
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameter', type=str, required=True,
                            help='parameter for which to measure bouts')
    parser.add_argument('--threshold', type=float, required=True,
                            help='threshold to define bouts')
    parser.add_argument('--greater_than', type=bool, required=False, default=True,
                            help='change to "less" to select bouts of low value')
    parser.add_argument('--force_matrix_size', type=int, required=False, default=0,
                            help='integer representing side length of transition matrix')
    
    args = parser.parse_args()
    greater_than = args.greater_than
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
    dataset = pd.DataFrame({'FlyID': [], 'group':[], 'kON':[], 'kOFF':[]})
    ID = []
    groups = []
    ONS = []
    OFFS = []
    for directory in glob.glob(DATADIR  + '*zoom*'):
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        try:
            fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        except:
            continue
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

        if (parameter + '_smoothed') not in fbf.columns:
            fbf[parameter] = fbf[parameter].fillna(method='pad', limit=5).fillna(fillValue)
            fbf = smooth(fbf, parameter)
            fbf.to_pickle(directory + '/frame_by_frame_synced.pickle')
        K =  np.diagonal(transition_matrix(binarize(fbf[parameter+'_smoothed'], threshold, greater_than)))[:]
        ID.append(FLY_ID)
        groups.append(GROUP)
        ONS.append(1.0-K[0])
        OFFS.append(1.0-K[1])
    dataset = pd.DataFrame({'FlyID': ID, 'group':groups, 'kON':ONS, 'kOFF':OFFS}) 
    for x in dataset.index:
        g =  dataset.loc[x, 'group'].split('-')
        dataset.loc[x, 'g1'] = g[0]
        dataset.loc[x, 'g2'] = g[1]
        dataset.loc[x, 'g3'] = g[2]

    dataset.to_pickle(DATADIR + parameter +'_'+str(threshold)+ '_transition_probabilities.pickle')
    dataset.to_csv(DATADIR + parameter +'_'+str(threshold)+ '_transition_probabilities.csv', sep=',')

    grouped = dataset.groupby(['g1','g3','FlyID'])
    lineplot(grouped.mean(), 'kON', 'Mean on transition')
    lineplot(grouped.mean(), 'kOFF', 'Mean off transition')
         
        
        
