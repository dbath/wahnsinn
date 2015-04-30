    
    
import pandas as pd
import flymad_jaaba.utilities as utilities
import glob
import numpy as np
from scipy import stats as st
from itertools import groupby
import matplotlib.pyplot as plt


FILEDIR = '/tier2/dickson/bathd/FlyMAD/JAABA_tracking/150427_pool_chrimshib/'
    
    
def get_first_poststim_bout(picklefile):
    jaaba_data = pd.read_pickle(picklefile)    
    poststim = jaaba_data[(jaaba_data.index > jaaba_data[(jaaba_data.Laser2_state + jaaba_data.Laser1_state) > 0.00001].index[-1])]
    #print picklefile.split('/')[-1], '\t', poststim.index[0]
    courting = poststim[(poststim.maxWingAngle >= 0.13)  & (poststim.dtarget <= 50 )]
    singing = poststim[poststim.maxWingAngle >= 0.13]
    targeting = poststim[poststim.dtarget <= 50]
    try:
        first_court = (courting.index[0] - poststim.index[0]).total_seconds()
    except:
        first_court = np.nan#600.0
        print 'no courtship found: ', picklefile.split('/')[-1]
    try:
        first_song = (singing.index[0] - poststim.index[0]).total_seconds()
    except:
        first_song = np.nan#480.0
        print 'no wingExt found: ', picklefile.split('/')[-1]
    try:
        first_targeting = (targeting.index[0] - poststim.index[0]).total_seconds()
    except:
        first_targeting = np.nan#600.0
        print 'no targeting found: ', picklefile.split('/')[-1]
        
    return first_court, first_song, first_targeting
  
  
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs)  



    
genotypes = []
latency_courting = []
latency_wExt = []
latency_target = []

for x in glob.glob(FILEDIR + 'JAR/*_fly.pickle'):
    fn = x.split('/')[-1].split('.pickle')[0]
    exp_id, DATE, TIME , _ = fn.split('_', 3)
    courtship, wingExt, targeting = get_first_poststim_bout(x)
    genotypes.append(exp_id)
    latency_courting.append(courtship)
    latency_wExt.append(wingExt)
    latency_target.append(targeting)


df = pd.DataFrame(zip(genotypes, latency_courting, latency_wExt, latency_target), columns=['Genotype', 'courting', 'wingExt', 'targeting'])

df.to_csv(FILEDIR + 'latencies.csv', sep=',')



for i in df.columns[1:]:
    means = df[i].groupby(df.Genotype).mean()
    ns = df[i].groupby(df.Genotype).count()
    sems = df[i].groupby(df.Genotype).std()

    bar_width = 0.5/(len(set(df.Genotype)))
    error_config = {'ecolor': '0.1'}
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    p_vals_gt = []
    p_vals_tr = []   
    
    for indx, klass in df[i].groupby(df.Genotype):
        xs = np.empty(len(list(klass.values)))
        xs.fill(sorted(set(df.Genotype)).index(indx)+1)
        jitter(xs, list(klass.values))


    p = plt.xlim(0,(len(set(df.Genotype))+1))

    q = plt.scatter(range(1,(len(set(df.Genotype))+1)), means, color = 'r', 
                                                           s = 50,
                                                           linewidth=2)
    (_, caps, _) = plt.errorbar(range(1,(len(set(df.Genotype))+1)), means, yerr=sems, linestyle='None',
                                                         elinewidth=1,
                                                         color='k',
                                                         capsize=25)
    for cap in caps:
        cap.set_color('k')
        cap.set_markeredgewidth(1)


    #Mann-Whitney test = st.ranksums
    """
    for j in sorted(set(df.Genotype)):
        position = (sorted(set(df.Genotype)).index(j))
        z_stat, p_val = st.ranksums(df[df.Genotype == j][i], 
                                    df[df.Genotype != j][i])

        q = plt.text(position+1,
                    means[position]+ sems[position] + 0.1*means.values.max(), #just above error bar
                    'p = '+ str(p_val.round(4)) + 
                    '\nn = ' + str(ns[position]),
                    size=10,
                    ha='center',
                    )

    """          
    ax.set_ylim(0,1.3*(max(df[i].dropna().values)))
    ax.set_ylabel('Latency (s)' + ' ' + u"\u00B1" + ' std')   # +/- sign is u"\u00B1"
    ax.set_xlabel(sorted(set(df.Genotype)))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #ax.legend(loc='upper right', shadow=False, fontsize=6)


    plt.show()
    fig.savefig(FILEDIR + i + '.svg', bbox_inches='tight')
        
