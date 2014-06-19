
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

STIM_ON  =  120          # set time of stimulus on in seconds
STIM_OFF =  180         #set time of stimulus off in seconds
BIN_SIZE =  20          # set bin size in seconds


paramlist = ['courtship',
            'courting (0)',
            'rayEllipseOrienting (0 -> 1)',
            'following (0 -> 1)',
            'wingExt (0)',
            'isOcclusionTouched',
            ]

INDEX_NAME = ['Courtship Index',
              'Courting Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Touching Index'
              ]
              
colourlist = ['g', 'k', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed', 'DarkGreen', 'DarkBlue', 'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']


# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)

"""
#for root, dirs, files in os.walk(DROP):
#    print rec
#    print root, dirs, files
for files in glob.glob(DROP):
    print rec, files
    for filename in fnmatch.filter(files, 'behavior.tsv'):
        print os.filename
        fullpath = os.path.basename(filename)
        print rec, fullpath
        FILE = pd.read_table(fullpath, sep='\t', index_col='File')
        print FILE
        pd.concat(rawframe, FILE)
    rec +=1
"""

rawfile = pd.read_table('/groups/dickson/home/bathd/Desktop/DROP/behavior.tsv', sep='\t', index_col='File' )    #READ FILE, indexed by filename """, index_col='File'"""
rawframe = DataFrame(rawfile)

qualityframe= rawframe[rawframe['quality'] > 0.8]    #REMOVE LOW QUALITY ARENAS
#print qualityframe['quality']
print 'removed bad arenas'
qualityflyframe = qualityframe[qualityframe['quality'] < 1]   #REMOVE EMPTY ARENAS
#print qualityflyframe['quality']
print 'removed empty arenas'
qualityframe= rawframe[rawframe['quality'] > 0.8]    #REMOVE LOW QUALITY ARENAS
#print qualityframe['quality']
print 'removed bad arenas'
qualityflyframe = qualityframe[qualityframe['quality'] < 1]   #REMOVE EMPTY ARENAS
#print qualityflyframe['quality']
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


# GENERATE PLOTS  #


fig = plt.figure()

for i in paramlist:
    cols = [col for col in averages.columns if i in col]
    cols = [col for col in cols if ', bin' in col]
    x = (np.arange(len(averages[cols].columns)))*BIN_SIZE + (BIN_SIZE)/2
    ys = averages[cols]   
    sems = semaverages[cols] 

    
    ax = fig.add_subplot(len(paramlist), 1 , 1+paramlist.index(i))
    #ax.set_color_cycle(colourlist)
    for j in ys.index:
        p = plt.plot(x, ys.ix[j], linewidth=2, label=j, zorder=100)
        plt.fill_between(x, ys.ix[j] + sems.ix[j], ys.ix[j] - sems.ix[j], alpha=0.05, zorder=90)
    ax.set_xlim((np.amin(x),np.amax(x)))
    ax.set_ylim(0,1.3*(ys.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)] + '+/- SEM')
    ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
    ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=10)
    #ax.set_title(i)

l = pl.legend(bbox_to_anchor=(0, 0, 0.95, 0.95), bbox_transform=pl.gcf().transFigure)

plt.show()
fig.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/courtship_by_event.svg", bbox_inches='tight')
        


print "Done."
