import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_bouts(binarySeries):
    
    d = binarySeries.fillna(method='pad', limit=3).dropna()
    t = d - d.shift()
    ts = t[abs(t) >0]
    ons = ts[ts[0]==1].index.values
    offs = ts[ts[0]==-1].index.values
    add_something=False
    #lons = list(ons)
    #loffs = list(offs)
    ontimes = []
    offtimes = []    
    for x in ons:
        ontimes.append(x)
    for x in offs:
        offtimes.append(x)
    
    if add_something == True:
        ontimes.insert(0,0)
    
    if len(ontimes) > len(offtimes):
        offtimes.append(t.index[-1])
    if len(offtimes) > len(ontimes):
        ontimes.insert(0,t.index[0])
    
    try:
        df = pd.DataFrame({'ons':ontimes, 'offs':offtimes})
        if len(df) == 0:
            df = pd.DataFrame({'ons':[0], 'offs':[0]})
            df['bout_duration'] = 0
            
        else:
            df['bout_duration'] = df['offs'] - df['ons']
            
    except:
        print len(ontimes), '\n\n\n', len(offtimes)
        print ontimes, '\n\n\n', offtimes
        df = pd.DataFrame({'ons':ons, 'offs':offs, 'bout_duration':np.nan})
    return df

def markov_chain_generator(_T, _P):
    x = np.zeros(_T)
    for t in range(_T-1):
        x[t+1] = np.random.rand() <= _P[x[t],x[t]]
    return x

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
    if 'max' in statistic:
        ax.set_ylabel(statistic + ' ' + statistic + ' ' + u"\u00B1" + ' SD', fontsize=20)   # +/- sign is u"\u00B1"
    else:
        ax.set_ylabel(statistic + ' ' + statistic + ' ' + u"\u00B1" + ' SEM', fontsize=20)
    ax.set_xticks(np.arange(0,len(means.index),1))
    ax.set_xticklabels(list(means.index), rotation=90)
    ax.tick_params(axis='y', labelsize=16 )
    ax.set_xlabel('Stimulus', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        
    plt.legend(prop={'size':10})           
    plt.show()
    #plt.savefig(DATADIR + args.parameter+ '_' + str(args.threshold) + '_'+statistic.replace(' ','_')+'_transition_probabilties.svg', bbox_inches='tight')
    return  


colourlist = ['#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
T = 10000
                                
#P = np.matrix([[0.994340, 0.00566],[0.786167,0.213833]])
#P = np.matrix([[0.213833,0.786167],[ 0.00566, 0.994340]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--transition_file', type=str, required=True,
                            help='directory of transition probabilities')  
                            
    args = parser.parse_args()
    trans = pd.read_pickle(args.transition_file )
    
    if 0: #turn this on to use a common transition matrix for each experimental group
        counts = trans.groupby(['g1','g3']).count()
        trans = trans.groupby(['g1','g3']).mean()
        trans = trans.reset_index()
        counts = counts.reset_index()
    
    for row in trans.index:
        P00 = 1.0 - trans.loc[row, 'kOFF']
        P10 = trans.loc[row, 'kOFF']
        P11 = 1.0 - trans.loc[row, 'kON']
        P01 = trans.loc[row, 'kON']
        P = np.matrix([[P11, P10],[P01,P00]]) #INVERTED TO FIT markov chain generator
        meanslist = []
        maxlist = []
        sumlist = []
        for i in range(1):#counts.loc[row, 'kON']):
            fake_data = markov_chain_generator(10000,P)
            bouts = get_bouts(pd.DataFrame(fake_data))
            meanslist.append(bouts.mean()['bout_duration'])
            maxlist.append(bouts.max()['bout_duration'])
            sumlist.append(bouts.sum()['bout_duration'])

        means = pd.Series(meanslist) / 15.0  #15 FPS converts to approximately per second.
        maxs = pd.Series(maxlist) / 15.0
        sums = pd.Series(sumlist) / 166.666666666667#convert to index per minute

        trans.loc[row, 'sim_mean'] = means.mean()
        trans.loc[row, 'sim_index'] = sums.mean() 
        trans.loc[row, 'sim_max'] = maxs.mean()
        #trans.loc[row, 'FlyID'] = row
    
    trans.to_pickle(args.transition_file.split('.pickle')[0] + '_simulation_data.pickle')
    lineplot(trans.groupby(['g1','g3','FlyID']).mean(), 'sim_index', 'Simulated index')
    lineplot(trans.groupby(['g1','g3','FlyID']).mean(), 'sim_mean', 'Simulated mean bout length')
    lineplot(trans.groupby(['g1','g3','FlyID']).mean(), 'sim_max', 'Simulated max bout length')
  
  
  
  
