import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas import Series






def fixitup(dfx):
    df = dfx.T
    df.index.name = 'Temperature'
    df = df.replace('Err:520', np.nan)
    df = df.astype(float)
    return df
    
    
def plot_from_csv(mean, sem):
    fig = plt.figure()
    #mean = mean.sort(columns = 'Temperature')
    #sem = sem.sort(columns = 'Temperature')
    #n = n.sort(columns = 'Temperature')
    
    error_config = {'ecolor': '0.1'}
        
    ax = fig.add_subplot(1,1,1)
        
    for j in mean.columns[:-1]:
        strx = list(mean[j].index.values)
        x = mean[j].index.values
        y = mean[j].values
        psems = sem[j].values
        nsems = (-1*(sem[j]))

        #top_errbar = tuple(map(sum, zip(psems, y)))
        #bottom_errbar = tuple(map(sum, zip(nsems, y)))
        q = plt.scatter(strx, y, 60, 
                        color = colourlist[list(mean.columns).index(j)], 
                        label = genotypes[list(mean.columns).index(j)], 
                        marker = 'o', 
                        zorder=200)

        p = plt.errorbar(x, y, xerr=None, yerr=psems,
                         linestyle='None' ,
                         fmt='',
                         ecolor='Gray',
                         elinewidth=1,
                         capsize=3,
                         zorder=100
                         ) 
       
        ax.set_xlim(24,33)     #HACK DANNO!!!!!
        #ax.set_xlim(means.index.values.min(), mean.index.values.max())
        #ax.set_ylim(0,1.3*(mean.values.max()))
        #ax.set_ylabel(INDEX_NAME[params.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
        ax.set_ylabel('Courtship Index ' + u"\u00B1" + ' SEM')
        ax.set_xlabel('Temperature (degrees C)')
        #ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=10)
        #ax.set_title(i)
        #ax.savefig(OUTPUT + i + '_vs_time.svg', bbox_inches='tight')

    l = plt.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=plt.gcf().transFigure)

    plt.show()
    fig.savefig('/groups/dickson/home/bathd/Dropbox/THESIS/X011_pIP10_adt2_co-activation/plot.svg', bbox_inches='tight')


colourlist = ['g', 'k', 'purple','k','g','r','c','m','y']

    
means = pd.read_csv('/groups/dickson/home/bathd/Dropbox/THESIS/X011_pIP10_adt2_co-activation/courtshipbytemperature_groupedbygenotype_means.csv',sep=',', index_col=0, header=0)
sems = pd.read_csv('/groups/dickson/home/bathd/Dropbox/THESIS/X011_pIP10_adt2_co-activation/courtshipbytemperature_groupedbygenotype_SEM.csv',sep=',', index_col=0, header=0)
#ns = pd.read_csv('/groups/dickson/home/bathd/Dropbox/THESIS/X009_P1_adt2_co-activation/ns.csv',sep=',', index_col=0, header=0)

means = fixitup(means)
sems = fixitup(sems)
#ns = fixitup(ns)

genotypes = ['aDT2', 'aDT2 & pIP10', 'pIP10']

plot_from_csv(means, sems)
print 'done'


