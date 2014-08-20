
import os, fnmatch
import glob
import re
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import Panel
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl
from scipy import stats as st
import pickle
import argparse



STIM_ON  =  0          # set time of stimulus on in seconds
STIM_OFF =  1         #set time of stimulus off in seconds
FPS =  25.0  


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='directory of Matebook Project files')  
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to store analysis')
parser.add_argument('--filedir', type=str, required=True,
                        help='directory of list of genotype information')
args = parser.parse_args()

_DROP = args.inputdir

OUTPUT = args.outputdir

if not os.path.exists(OUTPUT+ 'JAR'):
    os.makedirs(OUTPUT+ 'JAR')

JAR = OUTPUT+ 'JAR/'

_filefile = pd.read_csv(args.filedir+ 'Filenames.csv', sep=',')
for x in _filefile.index:
    _filefile['Filename'][x] = _filefile['Filename'][x].rsplit('/',2)[-1]
_filefile.index = _filefile['Filename']
_filefile.index.name = 'Video'



CONTROL_GENOTYPE = 'PooledControls'

CONTROL_TREATMENT = ''

groupinglist = ['Tester',
                'Target',
                'Time'
                ]

MFparamlist = [['Unnamed: 3_level_0', 'courtship'],
            ['0', 'courting'],
            ['0 1','rayEllipseOrienting'],
            ['0 1', 'following'],
            ['0', 'wingExt'],
            ['0', 'copulating'],
            ['0', 'movedAbs_u']
            ]

MMparamlist = [['Unnamed: 3_level_0', 'courtship'],
            ['1', 'courting'],
            ['1 0','rayEllipseOrienting'],
            ['1 0', 'following'],
            ['1', 'wingExt'],
            ['1', 'copulating'],
            ['1', 'movedAbs_u']
            ]
            
_paramlist = MFparamlist
params = []
for a,b in _paramlist:
    params.append(b)

INDEX_NAME = ['Courtship Index',
              'Courting Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Copulation Index',
              'Speed (mm/s)'
              ]
              
colourlist = ['k', 'DarkGreen', 'g','DarkBlue', 'b', 'r', 'c', 'm', 'y','Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed',   'CornflowerBlue']
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

def parse_filename(filename):
    vidname, flyID = filename.rsplit('/', 3)[-3:-1]
    #vidname = vidname.rsplit('.', 2)[0]
    return vidname, flyID

def parse_tempdf_name(path):
    filename = path.split('/')[-1]
    vidname = filename.split('_',3)[0]
    vidname = vidname.rsplit('.',2)[0]
    flyID = filename.split('_',3)[2]
    return vidname, flyID

    

def process_data(directory, paramlist):
    for filename in find_files(directory, 'track.tsv'):
        fi = pd.read_table(filename, sep='\t', header = [0,1], skiprows=[2,3])
        tempdf = DataFrame(index = fi.index)
        vidname, flyID = parse_filename(filename)
        tag = vidname + "_" + flyID
        if fi['Unnamed: 8_level_0', 'isMissegmented'].mean() >= 0.2:
            print "arena dropped for poor quality: ", tag
            continue
        elif fi['Unnamed: 8_level_0', 'isMissegmented'].mean() == 0.0:
            print "arena dropped because quality = 1: ", tag
            continue
        elif len(set(fi['Unnamed: 3_level_0', 'courtship'])) <=1:
            print "arena dropped because courtship = nan: ", tag
            continue
        else:
            for j in paramlist:
                tempdf[j[1]] = fi[j[0],j[1]]
        #tempdf.columns = [[vidname]*len(tempdf.columns),[flyID]*len(tempdf.columns),tempdf.columns]
        tempdf['Time'] = tempdf.index/FPS
        tempdf.to_pickle(JAR + tag + '_tempdf.pickle')
        print ".....", tag, " processed to pickling."
    return 

def retrieve_data(filename):
    
    rawfile = pd.read_pickle(JAR + 'rawfile.pickle')
    return rawfile
    
def compile_data(files):
    print 'compiling...'
    rawfile = DataFrame({'Time':[]})
    dflist = []
    vidlist = []
    flyIDlist = []
    for x in files:
        tempdf = pd.read_pickle(x)
        dflist.append(tempdf)
        vidname, flyID = parse_tempdf_name(x)
        vidlist.append(vidname)
        flyIDlist.append(flyID)
    rawfile = pd.concat(dflist,  keys=zip(vidlist,flyIDlist), names=['Video','Arena'])
    rawfile.to_csv(OUTPUT + 'rawfile.csv', sep=',')
    rawfile.to_pickle(JAR + 'rawfile.pickle')
    return rawfile

def group_data(rawfile, filefile):
    print "merging with fly info..."
    for x in filefile.columns:
        rawfile[x] = filefile[x].reindex(rawfile.index, level=0)
    #df = pd.merge(stretched_filefile, rawfile, left_index=True, right_index=True)
    df = rawfile
    df.to_csv(OUTPUT + 'df.csv', sep=',')
    print "grouping by genotype and condition..."
    grouped = df.groupby(groupinglist)
    mean = grouped.mean()
    std = grouped.std()
    n = grouped.count()
    sem = grouped.aggregate(lambda x: st.sem(x, axis=None))

    mean.to_csv(OUTPUT + 'mean.csv', sep=',')
    n.to_csv(OUTPUT + 'n.csv', sep=',')
    sem.to_csv(OUTPUT + 'sem.csv', sep=',')
    
    mean.to_pickle(JAR + 'mean.pickle')
    n.to_pickle(JAR + 'n.pickle')
    sem.to_pickle(JAR + 'sem.pickle')
    
    std.to_csv(OUTPUT + 'std.csv', sep=',')
    return mean, sem, n
    

def plot_from_track(mean, sem, n):#, p_vals):
    fig = plt.figure()
    for i in params:
        means = mean[i]
        sems = sem[i]
        ns = n[i]
        
        opacity = np.arange(0.5,1.0,(0.5/len(list_of_treatments)))
        index = np.arange(len(list_of_treatments))
        bar_width = 1.0/(len(list_of_genotypes))
        error_config = {'ecolor': '0.1'}
        
        ax = fig.add_subplot(len(params), 1 , 1+params.index(i))
        
    
        for j,k in means.groupby(level=[0,1]):
            bar_num = sorted(list_of_genotypes).index(j[0])
            index_num = list(list_of_treatments).index(j[1])
            x = list(means[j[0],j[1]].index)
            #print "JKL:     ", j, k, l, bar_num, colourlist[bar_num]
            #print list(means[j,k].index)
            #print means.ix[j,k]
            print "adding: ", j[0], j[1], "to ", i, " plot."
            p = plt.plot(x, means.ix[j[0],j[1]], linewidth=2, zorder=100,
                        linestyle = LINE_STYLE_LIST[index_num],
                        color=colourlist[bar_num],
                        label=[j[0],j[1]]) 
            q = plt.fill_between(x, means.ix[j[0],j[1]] + sems.ix[j[0],j[1]], means.ix[j[0],j[1]] - sems.ix[j[0],j[1]], 
                        alpha=0.05, 
                       zorder=90,
                       color=colourlist[bar_num],
                       )
        
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_ylim(0,1.3*(means.values.max()))
        ax.set_ylabel(INDEX_NAME[params.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
        ax.set_xlabel('Time (s)')
        ax.axvspan(STIM_ON, STIM_OFF,  facecolor='red', alpha=0.3, zorder=10)
        #ax.set_title(i)

    l = pl.legend(bbox_to_anchor=(0, 0, 0.95, 0.95), bbox_transform=pl.gcf().transFigure)

    plt.show()
    fig.savefig(OUTPUT + 'behaviour_vs_time.svg', bbox_inches='tight')


processed_filelist = glob.glob(OUTPUT + '*tempdf.pickle')
total_filelist = glob.glob(_DROP + '*/*/track.tsv')
        
if os.path.isfile(JAR+ 'mean.pickle') == True:
    print "Using pickled grouped data."
    mean =  pd.read_pickle(JAR+ 'mean.pickle')
    sem = pd.read_pickle(JAR+ 'sem.pickle')
    n = pd.read_pickle(JAR+ 'n.pickle')
    rawfile = retrieve_data(JAR + 'rawfile.pickle')
elif os.path.isfile(JAR+ 'rawfile.pickle') == True:
    print "Using pickled rawfile."
    rawfile = retrieve_data(JAR + 'rawfile.pickle')
    mean, sem, n = group_data(rawfile, _filefile)
elif (len(processed_filelist) < 1):
    print "Processing data from scratch"
    process_data(_DROP, _paramlist)
    rawfile = compile_data(glob.glob(JAR + '*tempdf.pickle'))
    mean, sem, n = group_data(rawfile, _filefile)
else: 
    print "Using pickled tempdf files."    
    rawfile = compile_data(processed_filelist)
    mean, sem, n = group_data(rawfile, _filefile)


list_of_treatments = set((_filefile[groupinglist[1]][_filefile[groupinglist[1]].notnull()]))
list_of_genotypes = set(_filefile[groupinglist[0]][_filefile[groupinglist[0]].notnull()])


p = plot_from_track(mean, sem, n)#, p_vals)

# GENERATE PLOTS  #


fig = plt.figure()




print "Done."
