
import os
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

groupinglist = ['Genotype',
                'Condition',
                'Treatment'
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
              
colourlist = ['k', 'g', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed', 'DarkGreen', 'DarkBlue', 'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']

LINE_STYLE_LIST = ['-', '--', '-.', ':']


# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)


rawfile = pd.read_table('/groups/dickson/home/bathd/Desktop/DROP/behavior.tsv', sep='\t', index_col='File' )    #READ FILE, indexed by filename """, index_col='File'"""


rawfile = rawfile[rawfile['quality'] > 0.8]    #REMOVE LOW QUALITY ARENAS
print 'removed bad arenas'
rawfile = rawfile[rawfile['quality'] < 1]      #REMOVE EMPTY ARENAS
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



# GENERATE PLOTS  #


fig = plt.figure()
fig.set_size_inches(3,18)

for i in paramlist:

    ys = mean[i]   
    sems = sem[i] 
    ns = n[i]
    opacity = np.arange(0.3,1,(1.0/len(set((df['Condition'][df['Condition'].notnull()])))))

    ax = fig.add_subplot(len(paramlist), 1 , 1+paramlist.index(i))
    for j in sorted(set(df['Condition'][df['Condition'].notnull()])):
        for k in sorted(set(df['Genotype'][df['Genotype'].notnull()])):
            vals = mean.xs((k,j), level=[0,1])
            yerr = sem.xs((k,j), level=[0,1])
            ax.errorbar(vals.index.values, vals[i],  
                        yerr=yerr[i],
                        fmt='-',
                        color = colourlist[sorted(set(df['Genotype'][df['Genotype'].notnull()])).index(k)],
                        alpha = 0.9,
                        linestyle = LINE_STYLE_LIST[sorted(set(df['Condition'][df['Condition'].notnull()])).index(j)],
                        linewidth = 2,
                        ecolor = 'k',
                        elinewidth = 0.5,
                        label=str(k) + ', ' + str(j) 
                        )

    plt.text(-0.3, 0.0, ns,
            size=4,
            transform=ax.transAxes,
            bbox=dict(facecolor='white'))
            
    
    ax.set_ylim(0,1.3*(ys.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
    ax.set_xlim(-500, 5000)
    ax.set_xlabel('Mass(ng) of cVA perfumed onto target')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend(loc='upper right', shadow=False, fontsize=6)


plt.show()
fig.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/means_sem_line.svg", bbox_inches='tight')
        


print "Done."
