
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



STIM_LIST = [[120,180],[360,420]]
FPS =  25.0  


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='directory of Matebook Project files')  
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to store analysis')
parser.add_argument('--filedir', type=str, required=True,
                        help='directory of list of genotype information')
parser.add_argument('--binsize', type=str, required=True,
                        help='numerical value, size of bins in seconds')
                       
                        
args = parser.parse_args()

_DROP = args.inputdir

OUTPUT = args.outputdir

binsize = float(args.binsize)

if not os.path.exists(OUTPUT+ 'JAR'):
    os.makedirs(OUTPUT+ 'JAR')

JAR = OUTPUT+ 'JAR/'

print args.filedir + 'Filenames.csv'
_filefile = pd.read_csv(args.filedir+ 'Filenames.csv', sep=',')
for x in _filefile.index:
    _filefile['Filename'][x] = _filefile['Filename'][x].rsplit('/',2)[-1]
    if '.MTS' in _filefile['Filename'][x]:
        _filefile['Filename'][x] = _filefile['Filename'][x].split('.MTS')[0] + '.MTS'
    if '.avi' in _filefile['Filename'][x]:
        _filefile['Filename'][x] = _filefile['Filename'][x].split('.avi')[0] + '.avi'
    
_filefile.index = _filefile['Filename']
_filefile.index.name = 'Video'



CONTROL_GENOTYPE = 'uas-Chrimson/+'

CONTROL_TREATMENT = ''

groupinglist = ['Tester',
                'Target'
                ]

MFparamlist = [['Unnamed: 3_level_0', 'courtship'],
            ['0 1','rayEllipseOrienting'],
            ['0 1', 'following'],
            ['0', 'wingExt'],
            ['0', 'copulating'],
            ['0', 'movedAbs_u']
            ]

MMparamlist = [['Unnamed: 3_level_0', 'courtship'],
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
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Copulation Index',
              'Speed (mm/s)'
              ]
              
colourlist = ['k','b', 'r', 'c', 'm', 'y','g', 'DarkGreen', 'DarkBlue', 'Orange', 'LightSlateGray', 'Indigo', 'GoldenRod', 'DarkRed',   'CornflowerBlue']
#colourlist = ['#000000','#0000FF', '#FF0000',  '#8EC8FF', '#999999' ,'#FF9966']
#colourlist = ['#008000','#0032FF','#000000','r','c','m','y']


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

def parse_tempdf_name(filename):
    if '.avi' in filename:
        vidname = filename.split('.avi')[0] + '.avi'
    if '.MTS' in filename:
        vidname = filename.split('.MTS')[0] + '.MTS'
    vidname = vidname.rsplit('/',1)[-1]
    flyID = filename.split('.mbr_')[1]
    flyID = flyID.rsplit('_',1)[0]
    return vidname, flyID

def parse_filename(path):
    vidname = path.split('.mbd/')[-1] 
    vidname = vidname.rsplit('.mbr')[0] + '.mbr'
    flyID = path.split('.mbr/')[1]
    flyID = flyID.split('/track.tsv',1)[0]
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
                if 'movedAbs_u' in j:
                    tempdf[j[1]] = tempdf[j[1]] * FPS
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
        print x
        tempdf = pd.read_pickle(x)
        dflist.append(tempdf)
        vidname, flyID = parse_tempdf_name(x)
        vidlist.append(vidname)
        flyIDlist.append(flyID)
    rawfile = pd.concat(dflist, keys=zip(vidlist,flyIDlist), names=['Video','Arena'])
    rawfile.to_csv(OUTPUT + 'rawfile.csv', sep=',')
    rawfile.to_pickle(JAR + 'rawfile.pickle')
    return rawfile

def bin_data(df):
    print "binning data to ", binsize, "s bins..."
    bins = np.arange(int(df['Time'].min()),int(df['Time'].max()),binsize)
    binned = df.groupby(groupinglist + ['Filename', 'Arena', pd.cut(df.Time, bins)])
    #binned.to_csv(OUTPUT + 'df_'+ args.binsize, 's_bins.csv', sep=',')
    return binned

def group_data(rawfile, filefile):
    print "merging with fly info..."
    for x in filefile.columns:
        rawfile[x] = filefile[x].reindex(rawfile.index, level=0)
    rawfile.to_pickle(JAR + 'df.pickle')
    rawfile.to_csv(OUTPUT + 'df.csv', sep=',')
    df = pd.read_csv(OUTPUT + 'df.csv', sep=',')  ###HACK danno.(include indices in column headers)
    binned = bin_data(df)
    print "grouping by genotype and condition..."
    grouped = binned.mean().groupby(level=(range(len(groupinglist)) + [-1]))
    mean = grouped.mean()
    std = grouped.std()
    n = grouped.count()
    sem = grouped.aggregate(lambda x: st.sem(x, axis=None))

    mean.to_csv(OUTPUT + 'mean_' + args.binsize + '.csv', sep=',')
    n.to_csv(OUTPUT + 'n_' + args.binsize + '.csv', sep=',')
    sem.to_csv(OUTPUT + 'sem_' + args.binsize + '.csv', sep=',')
    
    mean.to_pickle(JAR + 'mean_' + args.binsize + '.pickle')
    n.to_pickle(JAR + 'n_' + args.binsize + '.pickle')
    sem.to_pickle(JAR + 'sem_' + args.binsize + '.pickle')
    
    std.to_csv(OUTPUT + 'std_' + args.binsize + '.csv', sep=',')
    return mean, sem, n
    

def plot_from_track(mean, sem, n):#, p_vals):
    fig = plt.figure()
    mean = mean.sort(columns = 'Time')
    sem = sem.sort(columns = 'Time')
    n = n.sort(columns = 'Time')
    
    for i in params:
        means = mean[i]
        sems = sem[i]
        ns = n[i]
        times = mean['Time']
        
        opacity = np.arange(0.5,1.0,(0.5/len(list_of_treatments)))
        index = np.arange(len(list_of_treatments))
        bar_width = 1.0/(len(list_of_genotypes))
        error_config = {'ecolor': '0.1'}
        
        ax = fig.add_subplot(len(params), 1 , 1+params.index(i))
        
    
        for j,k in means.groupby(level=[0,1]):
            bar_num = sorted(list_of_genotypes).index(j[0])
            index_num = list(list_of_treatments).index(j[1])
            #x = list(times.ix[j[0],j[1]])
            #x = list(means[j[0],j[1]].index)
            x = list(times.ix[j[0]])
            y = list(means.ix[j[0]])
            psems = list(sems.ix[j[0]])
            nsems = list(-1*(sems.ix[j[0]]))
            top_errbar = tuple(map(sum, zip(psems, y)))
            bottom_errbar = tuple(map(sum, zip(nsems, y)))
            print "adding: ", j[0], j[1], "to ", i, " plot.", bar_num
            p = plt.plot(x, y, linewidth=2, zorder=100,
                        linestyle = LINE_STYLE_LIST[index_num],
                        color=colourlist[bar_num],
                        label=[j[0],j[1]]) 
            q = plt.fill_between(x, 
                                top_errbar, 
                                bottom_errbar, 
                                alpha=0.05, 
                                zorder=90,
                                color=colourlist[bar_num],
                                )
        
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_ylim(0,1.3*(means.values.max()))
        ax.set_ylabel(INDEX_NAME[params.index(i)] + ' ' + u"\u00B1" + ' SEM')   # +/- sign is u"\u00B1"
        ax.set_xlabel('Time (s)')
        for ons, offs in STIM_LIST:
            ax.axvspan(ons, offs,  facecolor='red', alpha=0.3, zorder=10)
        #ax.set_title(i)
        #ax.savefig(OUTPUT + i + '_vs_time.svg', bbox_inches='tight')

    l = pl.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=pl.gcf().transFigure)

    plt.show()
    fig.savefig(OUTPUT + 'behaviour_vs_time_' + args.binsize + '.svg', bbox_inches='tight')

def plot_bar_from_track(mean, sem, n):#, p_vals):
    fig = plt.figure()
    mean = mean.sort(columns = 'Time')
    sem = sem.sort(columns = 'Time')
    n = n.sort(columns = 'Time')
    
    for i in params:
        means = mean[i]
        sems = sem[i]
        ns = n[i]
        times = mean['Time']
        
        opacity = np.arange(0.5,1.0,(0.5/len(list_of_treatments)))
        index = np.arange(len(list_of_treatments))
        bar_width = 1.0/(len(list_of_genotypes))
        error_config = {'ecolor': '0.1'}
        
        p_vals_gt = []
        p_vals_tr = [] 
        
        ax = fig.add_subplot(len(params), 1 , 1+params.index(i))
        
        for j,k in means.groupby(level=[0,1]):
            bar_num = sorted(list_of_genotypes).index(j[0])
            index_num = list(list_of_treatments).index(j[1])    
                
            #x = list(times.ix[j[0]])
            #y = list(means.ix[j[0]].mean())
            #psems = list(sems.ix[j[0]].mean())
            #nsems = list(-1*(sems.ix[j[0]].mean()))
            #top_errbar = tuple(map(sum, zip(psems, y)))
            #bottom_errbar = tuple(map(sum, zip(nsems, y)))    
                
            p = plt.bar(0.1*(index_num+1)+index_num+(bar_width*bar_num), means.ix[j[0]].mean(), bar_width,
                    alpha=opacity[index_num],
                    color=colourlist[bar_num],
                    yerr=sems.ix[j[0]].mean(),
                    error_kw=error_config,
                    label=[j[0],j[1]])  
        
        
        
processed_filelist = glob.glob(JAR + '*tempdf.pickle')
total_filelist = glob.glob(_DROP + '*/*/track.tsv')

if os.path.isfile(JAR + 'mean_' + args.binsize + '.pickle') == True:
    print "Using pickled grouped data."
    mean =  pd.read_pickle(JAR+ 'mean_' + args.binsize + '.pickle')
    sem = pd.read_pickle(JAR+ 'sem_' + args.binsize + '.pickle')
    n = pd.read_pickle(JAR+ 'n_' + args.binsize + '.pickle')
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
