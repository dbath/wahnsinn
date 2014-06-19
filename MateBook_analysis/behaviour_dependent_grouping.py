
import os
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl

STIM_ON  =   120             # set time of stimulus on
STIM_OFF =  180         #set time of stimulus off

BIN_SIZE =  20          # set bin size in seconds

paramlist = ['rayEllipseOrienting (0 -> 1)',
            'following (0 -> 1)',
            'wingExt (0)',
            'isOcclusionTouched'
            ]

CATEGORIZATION_THRESHOLD = [0.25,
                            0.25,
                            0.20,
                            0.25
                            ]
CATEGORY_NAME = ['Orienter',
                 'Follower',
                 'Singer',
                 'Toucher'
                 ]    

INDEX_NAME = ['Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Touching Index'
              ]
           

colourlist = ['g', 'k', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed', 'DarkGreen', 'DarkBlue', 'CornflowerBlue']

#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']



rawfile = pd.read_table('/groups/dickson/home/bathd/Desktop/DROP/behavior.tsv', sep='\t', index_col='File' )    #READ FILE, indexed by filename """, index_col='File'"""
rawframe = DataFrame(rawfile)

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


fig1 = plt.figure()



# Set the default color cycle to colourlist 
matplotlib.rcParams['axes.color_cycle'] = colourlist
matplotlib.rc('axes', color_cycle=colourlist)

#GENERATE SUBPLOTS
nrows = (len(paramlist) / 2) + (len(paramlist) > 0)
print paramlist
for k in paramlist:
    categorizing_bin = [blah for blah in qualityflyframe2.columns if k in blah]
    categorizing_bin = [blah for blah in categorizing_bin if ', bin 5' in blah]      #BIN NUMBER DEPENDS ON MATEBOOK SETTINGS. BOO.
    active_set = qualityflyframe2[qualityflyframe2[categorizing_bin].values > CATEGORIZATION_THRESHOLD[paramlist.index(k)]]
    
    grouped = active_set.groupby(['Tester'])
    grouped.index = grouped['Tester']
    averages = grouped.mean()
    averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/averages_' + k + '.csv', sep=',')
    stdaverages = grouped.std()
    naverages = grouped.count()
    naverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n_' + k + '.csv', sep=',')
    naverages = naverages.drop(['Tester'], axis=1)
    sqrtnaverages = np.sqrt(naverages)
    semaverages = np.divide(stdaverages, sqrtnaverages)
    
    for i in paramlist:
        cols = [col for col in averages.columns if i in col]
        cols = [col for col in cols if ', bin' in col]
        x = (np.arange(len(averages[cols].columns)))*BIN_SIZE - (BIN_SIZE)/2
        #ys = [averages.ix[j,cols] for j in averages.index]
        ys = averages[cols]   
        sems = semaverages[cols] 
        ax = fig1.add_subplot(len(paramlist), len(paramlist), ((paramlist.index(k))*len(paramlist) + (paramlist.index(i) + 1)))
        #ax.set_color_cycle(colourlist)
        for j in ys.index:
            p = plt.plot(x, ys.ix[j], linewidth=2, label=j, zorder=100)
            plt.fill_between(x, ys.ix[j] + sems.ix[j], ys.ix[j] - sems.ix[j], alpha=0.05, zorder=90)
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
        ax.set_ylim(0,1.3*(ys.values.max()))
        ax.set_ylabel(INDEX_NAME[paramlist.index(i)])
        ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=3)
        ax.set_title(INDEX_NAME[paramlist.index(i)] + ' given ' + CATEGORY_NAME[paramlist.index(k)])

l = pl.legend(bbox_to_anchor=(0, 0, 0.9, 0.9), bbox_transform=pl.gcf().transFigure)
    

plt.show()
fig1.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/parameters_given_prestim_activity.svg", bbox_inches='tight')

fig2 = plt.figure()

for k in paramlist:
    categorizing_bin = [blah for blah in qualityflyframe2.columns if k in blah]
    categorizing_bin = [blah for blah in categorizing_bin if ', bin 5' in blah]      #BIN NUMBER DEPENDS ON MATEBOOK SETTINGS. BOO.
    active_set = qualityflyframe2[qualityflyframe2[categorizing_bin].values < CATEGORIZATION_THRESHOLD[paramlist.index(k)]]
    
    grouped = active_set.groupby(['Tester'])
    grouped.index = grouped['Tester']
    averages = grouped.mean()
    #averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/averages_' + k + '.csv', sep=',')
    stdaverages = grouped.std()
    naverages = grouped.count()
    #naverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n_' + k + '.csv', sep=',')
    naverages = naverages.drop(['Tester'], axis=1)
    sqrtnaverages = np.sqrt(naverages)
    semaverages = np.divide(stdaverages, sqrtnaverages)
    
    for i in paramlist:
        cols = [col for col in averages.columns if i in col]
        cols = [col for col in cols if ', bin' in col]
        x = (np.arange(len(averages[cols].columns)))*BIN_SIZE - (BIN_SIZE)/2
        #ys = [averages.ix[j,cols] for j in averages.index]
        ys = averages[cols]   
        sems = semaverages[cols] 
        ax = fig2.add_subplot(len(paramlist), len(paramlist), ((paramlist.index(k))*len(paramlist) + (paramlist.index(i) + 1)))
        #ax.set_color_cycle(colourlist)
        for j in ys.index:
            p = plt.plot(x, ys.ix[j], linewidth=2, label=j, zorder=100)
            plt.fill_between(x, ys.ix[j] + sems.ix[j], ys.ix[j] - sems.ix[j], alpha=0.05, zorder=90)
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
        ax.set_ylim(0,1.3*(ys.values.max()))
        ax.set_ylabel(INDEX_NAME[paramlist.index(i)])
        ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=3)
        ax.set_title(INDEX_NAME[paramlist.index(i)] + ' given not ' + CATEGORY_NAME[paramlist.index(k)])

l = pl.legend(bbox_to_anchor=(0, 0, 0.9, 0.9), bbox_transform=pl.gcf().transFigure)
    

plt.show()
fig2.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/parameters_given_prestim_activity_not.svg", bbox_inches='tight') 

#PLOT COURTERS VS NON-COURTERS

fig3 = plt.figure()
     
inactive_set = qualityflyframe2[qualityflyframe2['courting (0), bin 5'].values < 0.1] #BIN NUMBER DEPENDS ON MATEBOOK SETTINGS. BOO.
active_set = qualityflyframe2[qualityflyframe2['courting (0), bin 5'].values > 0.5] #BIN NUMBER DEPENDS ON MATEBOOK SETTINGS. BOO.

#courters:

grouped = active_set.groupby(['Tester'])
grouped.index = grouped['Tester']
averages = grouped.mean()
#averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/averages_' + k + '.csv', sep=',')
stdaverages = grouped.std()
naverages = grouped.count()
#naverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n_' + k + '.csv', sep=',')
naverages = naverages.drop(['Tester'], axis=1)
sqrtnaverages = np.sqrt(naverages)
semaverages = np.divide(stdaverages, sqrtnaverages)

for i in paramlist:
    cols = [col for col in averages.columns if i in col]
    cols = [col for col in cols if ', bin' in col]
    x = (np.arange(len(averages[cols].columns)))*BIN_SIZE - (BIN_SIZE)/2
    #ys = [averages.ix[j,cols] for j in averages.index]
    ys = averages[cols]   
    sems = semaverages[cols] 
    ax = fig3.add_subplot(len(paramlist), 2, 1+paramlist.index(i)*2)
    #ax.set_color_cycle(colourlist)
    for j in ys.index:
        p = plt.plot(x, ys.ix[j], linewidth=2, label=j, zorder=100)
        plt.fill_between(x, ys.ix[j] + sems.ix[j], ys.ix[j] - sems.ix[j], alpha=0.05, zorder=90)
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
        ax.set_ylim(0,1.3*(ys.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)])
    ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=3)
    ax.set_title(INDEX_NAME[paramlist.index(i)] + ' given high courtship before stim')

#non-courters:

grouped = inactive_set.groupby(['Tester'])
grouped.index = grouped['Tester']
averages = grouped.mean()
#averages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/averages_' + k + '.csv', sep=',')
stdaverages = grouped.std()
naverages = grouped.count()
#naverages.to_csv('/groups/dickson/home/bathd/Desktop/OUTPUT/n_' + k + '.csv', sep=',')
naverages = naverages.drop(['Tester'], axis=1)
sqrtnaverages = np.sqrt(naverages)
semaverages = np.divide(stdaverages, sqrtnaverages)

for i in paramlist:
    cols = [col for col in averages.columns if i in col]
    cols = [col for col in cols if ', bin' in col]
    x = (np.arange(len(averages[cols].columns)))*BIN_SIZE - (BIN_SIZE)/2
    #ys = [averages.ix[j,cols] for j in averages.index]
    ys = averages[cols]   
    sems = semaverages[cols] 
    ax = fig3.add_subplot(len(paramlist), 2, 2+paramlist.index(i)*2)
    #ax.set_color_cycle(colourlist)
    for j in ys.index:
        p = plt.plot(x, ys.ix[j], linewidth=2, label=j, zorder = 100)
        plt.fill_between(x, ys.ix[j] + sems.ix[j], ys.ix[j] - sems.ix[j], alpha=0.05, zorder=90)
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_xlabel('Time (s), binned to '+ str(BIN_SIZE) + 's')
        ax.set_ylim(0,1.3*(ys.values.max()))
    ax.set_ylabel(INDEX_NAME[paramlist.index(i)])
    ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=3)
    ax.set_title(INDEX_NAME[paramlist.index(i)] + ', given low courtship before stim')

l = pl.legend(bbox_to_anchor=(0, 0, 0.9, 0.9), bbox_transform=pl.gcf().transFigure)
    

plt.show()
fig3.savefig("/groups/dickson/home/bathd/Desktop/OUTPUT/parameters_given_non_courter.svg", bbox_inches='tight') 


print "Done."
