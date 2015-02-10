
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
import flymad_jaaba.utilities as utilities


FPS =  25.0  


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='directory of Matebook Project files')
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory of Matebook Project files')   
parser.add_argument('--binsize', type=str, required=True,
                        help='numerical value, size of bins in seconds')
parser.add_argument('--assay', type=str, required=False, default='MF',
                        help='Which assay: MF, MM, or M')                   
parser.add_argument('--email', type=str, required=False, default=None,
                        help='email address for notification (optional)')                  
args = parser.parse_args()

INPUT_DIR = args.inputdir
OUTPUT_DIR = args.outputdir
if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES'):
    os.makedirs(OUTPUT_DIR + '/SUMMARY_FILES')
if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES/PNG'):
    os.makedirs(OUTPUT_DIR + '/SUMMARY_FILES/PNG')
if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES/PDF'):
    os.makedirs(OUTPUT_DIR + '/SUMMARY_FILES/PDF')
if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES/SVG'):
    os.makedirs(OUTPUT_DIR + '/SUMMARY_FILES/SVG')

binsize_str = args.binsize + 's'
binsize = float(args.binsize)


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
            
Mparamlist = [['0','rightWingAngle'],
            ['0','leftWingAngle'],
            ['0', 'movedAbs_u'],
            ['0', 'turnedAbs']]




INDEX_PAIRS = ['Courtship Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Copulation Index',
              'Speed (mm/s)'
              ]

INDEX_SINGLES = ['Wing Ext. Angle (right)',
                 'Wing Ext. Angle (left)',
                 'Speed (mm/s)',
                 'Rotation']

if args.assay == 'M':
    _paramlist = Mparamlist
    INDEX_NAME = INDEX_SINGLES
    __STIM_LIST = [[30,35], [40,45],[50,55],[60,65],[70,75],[80,85],[90,95],[100,105],[110,115],[120,125],[130,135], [140,145],[150,155],[160,165],[170,175]]
    STIM_LIST = [[60,70],[80,90],[100,110],[120,130],[140,150],[160,170],[180,190],[200,210],[220,230],[240,250],[260,270],[280,290],
                [300,330],[360,390],[420,450],[480,510],[540,570],[600,630],[660,690],[720,750],[780,810],[840,870]]
if args.assay == 'MF':
    _paramlist = MFparamlist
    INDEX_NAME = INDEX_PAIRS 
    STIM_LIST = [[120,180],[360,420]]   
if args.assay == 'MM':
    _paramlist = MMparamlist
    INDEX_NAME = INDEX_PAIRS
    STIM_LIST = [[120,180],[360,420]] 
    
params = []
for a,b in _paramlist:
    params.append(b)

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def parse_tempdf_name(filename):
    filename = filename.split('/')[-1]
    vidname, flyID = filename.split('_',1)
    flyID = flyID.rsplit('_',1)[0]
    return vidname, flyID

def parse_filename(path):
    vidname = path.split('.mbd/')[-1] 
    vidname = vidname.rsplit('.mbr')[0] + '.mbr'
    flyID = path.split('.mbr/')[1]
    flyID = flyID.split('/track.tsv',1)[0]
    return vidname, flyID

def parse_screen_filename(path):
    exp_id = path.split('/')[-1].split('_')[0]
    return exp_id
    

def process_data(directory, paramlist, storage_location):
    for filename in find_files(directory, 'track.tsv'):
        fi = pd.read_table(filename, sep='\t', header = [0,1], skiprows=[2,3])
        tempdf = DataFrame(index = fi.index)
        vidpath, flyID = parse_filename(filename)
        vidname = parse_screen_filename(directory)
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
        tempdf.to_pickle(storage_location + '/'+ tag + '_arena.pickle')
        print ".....", tag, " processed to pickling."
    return 


    
def compile_data(pickle_jar):
    print 'compiling...'
    rawfile = DataFrame({'Time':[]})
    dflist = []
    vidlist = []
    flyIDlist = []
    for x in glob.glob(pickle_jar + '/*arena.pickle'):
        tempdf = pd.read_pickle(x)
        dflist.append(tempdf)
        vidname, flyID = parse_tempdf_name(x)
        vidlist.append(vidname)
        flyIDlist.append(flyID)
    rawfile = pd.concat(dflist, keys=flyIDlist, names=['Arena'])
    rawfile = rawfile.reset_index()
    #rawfile.to_csv(OUTPUT + 'rawfile.csv', sep=',')
    rawfile.to_pickle(pickle_jar + '/' + vidname + '_compiled.pickle')
    return rawfile

def bin_data(df):
    print "binning data to ", binsize, "s bins..."
    bins = np.arange(int(df['Time'].min()),int(df['Time'].max()),binsize)
    binned = df.groupby([ 'Arena', pd.cut(df.Time, bins)])
    return binned

def combine_annotations(rawfile, filefile):
    print "merging with fly info..."
    for x in filefile.columns:
        rawfile[x] = filefile[x].reindex(rawfile.index, level=0)
    rawfile.to_pickle(JAR + 'df.pickle')
    rawfile.to_csv(OUTPUT + 'df.csv', sep=',')
    return

def group_data(rawfile):
    binned = bin_data(rawfile)
    grouped = binned.mean().groupby(['Time'])
    
    mean = grouped.mean().reset_index()
    std = grouped.std().reset_index()
    n = grouped.count().reset_index()
    sem = grouped.aggregate(lambda x: st.sem(x, axis=None)).reset_index()

    #mean.to_csv(OUTPUT + 'mean_' + args.binsize + '.csv', sep=',')
    #n.to_csv(OUTPUT + 'n_' + args.binsize + '.csv', sep=',')
    #sem.to_csv(OUTPUT + 'sem_' + args.binsize + '.csv', sep=',')
    
    #mean.to_pickle(JAR + 'mean_' + args.binsize + '.pickle')
    #n.to_pickle(JAR + 'n_' + args.binsize + '.pickle')
    #sem.to_pickle(JAR + 'sem_' + args.binsize + '.pickle')
    
    #std.to_csv(OUTPUT + 'std_' + args.binsize + '.csv', sep=',')
    return mean, sem, n
    

def plot_from_track(mean, sem, n, exp_id, storage_location, timestamp):#, p_vals):
    print 'plotting: ', exp_id
    fig = plt.figure(figsize=(8.5,11), dpi=300, facecolor='w')
    fig.suptitle('SUMMARY: ' + exp_id + '\n' + timestamp, fontsize=16)
    mean = mean.sort(columns = 'Time')
    sem = sem.sort(columns = 'Time')
    n = n.sort(columns = 'Time')
    
    for i in params:
        means = mean[i]
        sems = sem[i]
        ns = n[i]
        times = mean['Time']
        
        error_config = {'ecolor': '0.1'}
        
        ax = fig.add_subplot(len(params)/2, 2 , 1+params.index(i))
        x = list(times)
        y = list(means)
        psems = list(sems)
        nsems = list(-1*(sems))
        top_errbar = tuple(map(sum, zip(psems, y)))
        bottom_errbar = tuple(map(sum, zip(nsems, y)))
        p = plt.plot(x, y, linewidth=2, zorder=100,
                    color='b',) 
        q = plt.fill_between(x, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.05, 
                            zorder=90,
                            color='b',
                            )
       
        ax.set_xlim((np.amin(x),np.amax(x)))
        ax.set_ylim(0,1.3*(means.values.max()))
        ax.set_ylabel(INDEX_NAME[params.index(i)] + ' ' + u"\u00B1" + ' SEM', fontsize=10)   # +/- sign is u"\u00B1"
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        for ons, offs in STIM_LIST:
            ax.axvspan(ons, offs,  facecolor='red', alpha=0.3, zorder=10)
        #ax.set_title(i)
        #ax.savefig(OUTPUT + i + '_vs_time.svg', bbox_inches='tight')
    #l = pl.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=pl.gcf().transFigure)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.87, wspace=0.40, hspace=0.32)
    #plt.show()
    fig.savefig(storage_location + '/SVG/' + exp_id + '_' + binsize_str + '.svg', bbox_inches='tight')
    fig.savefig(storage_location + '/PDF/' + exp_id + '_' + binsize_str + '.pdf', bbox_inches='tight')
    fig.savefig(storage_location + '/PNG/' + exp_id + '_' + binsize_str + '.png', bbox_inches='tight')
    plt.close(fig)

experiment_list=[]
for x in glob.glob(INPUT_DIR + '*.mbr'):
    exp_id = parse_screen_filename(x)
    if exp_id in experiment_list:
        continue
    else:
        experiment_list.append(exp_id)
        
for experiment in experiment_list:
    if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES/PNG/' + experiment+ '_' + binsize_str + '.png'):
        storage_location = OUTPUT_DIR + 'JAR/' + experiment
        if not os.path.exists(storage_location):
            os.makedirs(storage_location)
        for directory in glob.glob(INPUT_DIR + experiment + '*'):
            exp_id, timestamp = parse_tempdf_name(directory)
            process_data(directory, _paramlist, storage_location)
        
        compiled = compile_data(storage_location)
        mean, sem, n = group_data(compiled)
        plot_from_track(mean, sem, n, experiment, OUTPUT_DIR + '/SUMMARY_FILES/', timestamp)
 
utilities.sendMail(args.email, 'Matebook processing finished', ('Your files are available at: ' + OUTPUT_DIR))
print "Done."
