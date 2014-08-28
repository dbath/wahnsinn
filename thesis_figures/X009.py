
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


DROP = '/groups/dickson/home/bathd/Desktop/DROP/'   #Location of behavior.tsv and filenames.csv files

CONTROL_GENOTYPE = '+/UAS>>Kir2.1'

CONTROL_TREATMENT = 'male'

groupinglist = ['Tester',
                'Target',
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
            
paramlist = MMparamlist

INDEX_NAME = ['Courtship Index',
              'Courting Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index'
              ]
              
colourlist = ['k', 'g', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed', 'DarkGreen', 'DarkBlue', 'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']

LINE_STYLE_LIST = ['-', '--', '-.', ':']


# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)


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
    print 'Found Matebook Data:', filename


"""
rawfile = pd.read_table('/groups/dickson/home/bathd/Desktop/DROP/behavior.tsv', sep='\t', index_col='File' )    #READ FILE, indexed by filename , index_col='File'
"""

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
#sem = grouped.aggregate(lambda x: np.std(x, axis=1)/np.sqrt(x.count()))

mean.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/mean.csv', sep=',')
n.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n.csv', sep=',')
sem.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/sem.csv', sep=',')

std.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/std.csv', sep=',')

list_of_treatments = set((df[groupinglist[1]][df[groupinglist[1]].notnull()]))
list_of_genotypes = set(df[groupinglist[0]][df[groupinglist[0]].notnull()])


# GENERATE PLOTS  #


fig = plt.figure()
fig.set_size_inches(3,18)

for i in paramlist:

    means = mean[i] 
    sems = sem[i] 
    ns = n[i]

    opacity = np.arange(0.5,1.0,(0.5/len(list_of_treatments)))
    index = np.arange(len(list_of_treatments))
    bar_width = 1.0/(len(list_of_genotypes))
    error_config = {'ecolor': '0.1'}
    
    ax = fig.add_subplot(len(paramlist), 1 , 1+paramlist.index(i))
    p_vals_gt = []
    p_vals_tr = []    
    for (j,k), l in grouped:
        bar_num = sorted(list_of_genotypes).index(j)
        index_num = list(list_of_treatments).index(k)
        
        p = plt.bar(0.1*(index_num+1)+index_num+(bar_width*bar_num), means[j,k], bar_width,
                    alpha=opacity[index_num],
                    color=colourlist[bar_num],
                    yerr=sems[j,k],
                    error_kw=error_config,
                    label=[j,k])  
        #Mann-Whitney test = st.ranksums
        
        z_stat_gt, p_val_gt = st.ranksums(df[(df[groupinglist[0]] == j) & (df[groupinglist[1]] == k)][i], 
                                        df[(df[groupinglist[0]] == CONTROL_GENOTYPE) & (df[groupinglist[1]] == k)][i]) 
        z_stat_tr, p_val_tr = st.ranksums(df[(df[groupinglist[0]] == j) & (df[groupinglist[1]] == k)][i], 
                                        df[(df[groupinglist[0]] == j) & (df[groupinglist[1]] == CONTROL_TREATMENT)][i]) 

        p_vals_gt_rounded = [ '%.4f' % elem for elem in p_vals_gt]
        p_vals_tr_rounded = [ '%.4f' % elem for elem in p_vals_tr]
        
        q = plt.text(0.1*(index_num+1)+index_num+(bar_width*(bar_num + 0.5)), #centre of bar
                    means[j,k]+ 0.5*sems[j,k] + 0.1*means.values.max(), #just above error bar
                    'p(genotype) = '+ str(p_val_gt.round(4)) + 
                    '\np(treatment) = ' + str(p_val_tr.round(4)) + 
                    '\nn = ' + str(ns[j,k]),
                    size=10,
                    ha='center',
                    )

            
    
    ax.set_ylim(0,1.3*(means.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
    ax.set_xticks(0.1*(index+1)+index+0.5) # (bar_width*(len(list_of_genotypes)/2)))
    ax.set_xticklabels(list(list_of_treatments))
    ax.set_xlabel(groupinglist[1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend(loc='upper right', shadow=False, fontsize=6)


plt.show()
fig.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/grouped_means_sem_bar.svg", bbox_inches='tight')
        


print "Done."
