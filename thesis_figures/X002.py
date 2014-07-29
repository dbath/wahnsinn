
import os
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl


DROP = '/groups/dickson/home/bathd/Desktop/DROP/'   #Location of behavior.tsv and filenames.csv files


paramlist = ['courtship',
            'courting (1)',
            'rayEllipseOrienting (1 -> 0)',
            'following (1 -> 0)',
            'wingExt (1)'
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
INDEX_NAME = ['Courtship Index',
              'Courting Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index'
              ]
              
colourlist = ['k', 'g', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed', 'DarkGreen', 'DarkBlue', 'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']


# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)


rawfile = pd.read_table('/groups/dickson/home/bathd/Desktop/DROP/behavior.tsv', sep='\t', index_col='File' )    #READ FILE, indexed by filename """, index_col='File'"""
rawframe = DataFrame(rawfile)

qualityframe= rawframe[rawframe['quality'] > 0.8]    #REMOVE LOW QUALITY ARENAS
print 'removed bad arenas'
qualityflyframe = qualityframe[qualityframe['quality'] < 1]   #REMOVE EMPTY ARENAS
print 'removed empty arenas'


#GET LIST OF GENOTYPES FROM FILENAMES.CSV
filefile = pd.read_csv('/groups/dickson/home/bathd/Desktop/DROP/Filenames.csv', index_col='Filename', sep=',')
fileframe = DataFrame(filefile)  #column index is first row, row index is filename
genotype = DataFrame(fileframe['Tester'])
qualityflyframe2 = pd.merge(genotype, qualityflyframe, left_index=True, right_index=True)
qualityflyframe2.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/qff2.csv', sep=',')
grouped = qualityflyframe2.groupby(['Tester'])
grouped.index = grouped['Tester']


averages = grouped.mean()
stdaverages = grouped.std()
naverages = grouped.count()
averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/averages.csv', sep=',')
naverages = naverages.drop(['Tester'], axis=1)
naverages = naverages.drop(['Arena'], axis=1)
sqrtnaverages = np.sqrt(naverages)
semaverages = np.divide(stdaverages, sqrtnaverages)


averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/all_parameter_mean.csv', sep=',')
naverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/all_parameter_n.csv', sep=',')
semaverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/all_parameter_SEM.csv', sep=',')

blah = Series(averages.index)
print blah

# GENERATE PLOTS  #


fig = plt.figure()
fig.set_size_inches(3,18)

for i in paramlist:

    ys = averages[i]   
    sems = semaverages[i] 
    n = naverages[i]
    index = np.arange(len(ys)) #+ 0.2
    bar_width = .8
    opacity = 0.8
    error_config = {'ecolor': '0.1'}

    
    ax = fig.add_subplot(len(paramlist), 1 , 1+paramlist.index(i))
    #ax.set_color_cycle(colourlist)
    p = plt.bar(index, ys, bar_width,
                alpha=opacity,
                color=colourlist,
                yerr=sems,
                error_kw=error_config,
                label=ys.index.values)
    plt.text(0.05, 0.75, str(n),
            size='x-small',
            transform=ax.transAxes,
            bbox=dict(facecolor='white'))
            
    ax.set_ylim(0,1.3*(ys.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
    ax.set_xlim(-0.2,index.max()+bar_width+0.2)
    plt.xticks(index+bar_width/2., ys.index, fontsize=8)
    #ax.set_xticklabels(index+bar_width/2., ys.index)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #ax.set_xlabel('Genotype')
    #ax.set_title(i)
#plt.legend(ys[0], blah)

#l = pl.legend(bbox_to_anchor=(0, 0, 0.95, 0.95), bbox_transform=pl.gcf().transFigure)

plt.show()
fig.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/means_sem_bar.svg", bbox_inches='tight')
        


print "Done."
