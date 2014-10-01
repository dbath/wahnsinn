
import os, fnmatch
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl
from scipy import stats as st



STIM_ON  =  120          # set time of stimulus on in seconds
STIM_OFF =  180         #set time of stimulus off in seconds
BIN_SIZE =  10          # set bin size in seconds

DROP = '/groups/dickson/home/bathd/Desktop/DROP/'   #Location of behavior.tsv and filenames.csv files

CONTROL_GENOTYPE = '+/UAS-CsChrimson'

CONTROL_TREATMENT = '0.5Hz'

groupinglist = ['Genotype',
                'Stimulus Frequency',
                ]

MFparamlist = ['courtship',
            'courting (0)',
            'rayEllipseOrienting (0 -> 1)',
            'following (0 -> 1)',
            'wingExt (0)'
            ]

MMparamlist = ['courtship',
            'courting (1)',
            'rayEllipseOrienting (1 -> 0)',
            'following (1 -> 0)',
            'wingExt (1)'
            ]
            
paramlist = MFparamlist

INDEX_NAME = ['Courtship Index',
              'Courting Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index'
              ]
              
colourlist = ['k','r','Orange', 'b', 'g', 'DarkGreen', 'g','DarkBlue', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed',   'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']

LINE_STYLE_LIST = ['-', '--', '-.', ':']


# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)

def set_column_sequence(dataframe, seq):
    '''Takes a dataframe and a subsequence of its columns, returns dataframe with seq as first columns'''
    cols = seq[:] # copy so we don't mutate seq
    for x in dataframe.columns:
        if x not in cols:
            cols.append(x)
    return dataframe[cols]
    
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

rawfile = DataFrame()

for filename in find_files(DROP, 'behavior.tsv'):
    fi = pd.read_table(filename, sep='\t', index_col='File')
    rawfile = pd.concat([rawfile, fi])
    rawfile = set_column_sequence(rawfile, fi.columns)
    print 'Found Matebook Data:', filename


rawfile = rawfile[rawfile['quality'] > 0.8]    #REMOVE LOW QUALITY ARENAS
print 'removed bad arenas'
rawfile = rawfile[rawfile['quality'] < 1]      #REMOVE EMPTY ARENAS
rawfile = rawfile[np.isfinite(rawfile['courtship'])]  #REMOVE NON-ARENAS
print 'removed empty arenas'


#GET LIST OF GENOTYPES FROM FILENAMES.CSV
filefile = pd.read_csv('/groups/dickson/home/bathd/Desktop/DROP/Filenames.csv', index_col='Filename', sep=',')

df = pd.merge(filefile, rawfile, left_index=True, right_index=True)
df.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/rawfile.csv', sep=',')

grouped = df.groupby(groupinglist)
mean = grouped.mean()
std = grouped.std()
n = grouped.count()
sem = grouped.aggregate(lambda x: st.sem(x, axis=None))

mean.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/mean.csv', sep=',')
n.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n.csv', sep=',')
sem.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/sem.csv', sep=',')

std.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/std.csv', sep=',')

list_of_treatments = set((df[groupinglist[1]][df[groupinglist[1]].notnull()]))
list_of_genotypes = set(df[groupinglist[0]][df[groupinglist[0]].notnull()])


# GENERATE PLOTS  #


fig = plt.figure()

for i in paramlist:
    cols = [col for col in mean.columns if i in col]
    cols = [col for col in cols if ', bin' in col]
    x = (np.arange(len(mean[cols].columns)))*BIN_SIZE + (BIN_SIZE)/2
    means = mean[cols]   
    sems = sem[cols] 
    ns = n[cols]
    opacity = np.arange(0.5,1.0,(0.5/len(list_of_treatments)))
    index = np.arange(len(list_of_treatments))
    bar_width = 1.0/(len(list_of_genotypes))
    error_config = {'ecolor': '0.1'}
    #print "MAX: ", i, "--- \t",  np.amax(means)
    ax = fig.add_subplot(len(paramlist), 1 , 1+paramlist.index(i))
    
    for (j,k), l in grouped:
        bar_num = sorted(list_of_genotypes).index(j)
        index_num = list(list_of_treatments).index(k)
        p = plt.plot(x, means.ix[j,k], linewidth=2, zorder=100,
                    linestyle = LINE_STYLE_LIST[index_num],
                    color=colourlist[bar_num],
                    label=[j,k]) 
        q = plt.fill_between(x, means.ix[j,k] + sems.ix[j,k], means.ix[j,k] - sems.ix[j,k], 
                    alpha=0.05, 
                    zorder=90,
                    color=colourlist[bar_num]
                    )
    
    ax.set_xlim((np.amin(x),np.amax(x)))
    ax.set_ylim(0,0.5)#1.3*(means.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
    ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
    ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=10)
    #ax.set_title(i)

l = pl.legend(bbox_to_anchor=(0, 0, 0.95, 0.95), bbox_transform=pl.gcf().transFigure)

plt.show()
fig.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/behaviour_vs_time.svg", bbox_inches='tight')
        


print "Done."
