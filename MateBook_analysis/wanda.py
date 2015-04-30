
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
import matplotlib.collections as collections
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pylab as pl
from scipy import stats as st
import pickle
import argparse
import flymad_jaaba.utilities as utilities

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--matebookdir', type=str, required=True,
                        help='directory of Matebook Project files')
parser.add_argument('--rawdatadir', type=str, required=True,
                        help='directory of videos, metadata, etc')                          
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to save final files')   
parser.add_argument('--binsize', type=str, required=False, default='30',
                        help='numerical value, size of bins in seconds, default: 30s')
parser.add_argument('--assay', type=str, required=False, default='MF',
                        help='Which assay: MF, MM, or M')                   
parser.add_argument('--email', type=str, required=False, default=None,
                        help='email address for notification (optional)')  
parser.add_argument('--framerate', type=float, required=False, default=25.0,
                        help='enter framerate, default: 25.0')     
parser.add_argument('--thresholdtemperature', type=float, required=False, default=26.0,
                        help='enter threshold temperature for cumulative plots, default: 26.0')    
parser.add_argument('--plots_only', type=str, required=False, default=False,
                        help='make True to skip data processing and plot.') 
                                        
args = parser.parse_args()

INPUT_DIR = args.matebookdir
RAWDATA_DIR = args.rawdatadir
OUTPUT_DIR = args.outputdir
ASSAY = args.assay
THRESHOLD = args.thresholdtemperature
plots_only = args.plots_only

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

FPS = args.framerate



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
            ['1', 'courting'],
            ['1', 'movedAbs_u']
            ]
            
Mparamlist = [['0','rightWingAngle'],
            ['0','leftWingAngle'],
            ['0', 'movedAbs_u'],
            ['0', 'turnedAbs']]




MMINDEX_PAIRS = ['Courtship Index',
              'Orienting Index',
              'Following Index',
              'Wing Ext. Index',
              'Courting Index',
              'Speed (mm/s)'
              ]
MFINDEX_PAIRS = ['Courtship Index',
                'Orienting Index',
                'Following Index',
                'Wing Ext. Index',
                'Copulating Index',
                'Speed (mm/s)'
                ]

INDEX_SINGLES = ['Wing Ext. Angle (right)',
                 'Wing Ext. Angle (left)',
                 'Speed (mm/s)',
                 'Rotation']


if ASSAY == 'M':
    _paramlist = Mparamlist
    INDEX_NAME = INDEX_SINGLES
    __STIM_LIST = [[30,35], [40,45],[50,55],[60,65],[70,75],[80,85],[90,95],[100,105],[110,115],[120,125],[130,135], [140,145],[150,155],[160,165],[170,175]]
    _STIM_LIST = [[60,70],[80,90],[100,110],[120,130],[140,150],[160,170],[180,190],[200,210],[220,230],[240,250],[260,270],[280,290],
                [300,330],[360,390],[420,450],[480,510],[540,570],[600,630],[660,690],[720,750],[780,810],[840,870]]
    STIM_LIST = []
if ASSAY == 'MF':
    _paramlist = MFparamlist
    INDEX_NAME = MFINDEX_PAIRS 
    STIM_LIST = [[120,180],[360,420]]   
if ASSAY == 'MM':
    _paramlist = MMparamlist
    INDEX_NAME = MMINDEX_PAIRS
    _STIM_LIST = [[120,180],[360,420]] 
    STIM_LIST = []

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
    exp_id = path.split('/')[-1].split('.')[0]#.split('_')[0]
    return exp_id
    

def process_matebook_data(directory, paramlist, storage_location):
    vidname = parse_screen_filename(directory)
    for filename in find_files(directory, 'track.tsv'):
        vidpath, flyID = parse_filename(filename)
        tag = vidname + "_" + flyID
        if not os.path.exists(storage_location + '/' + tag + '_arena.pickle'):
            fi = pd.read_table(filename, sep='\t', header = [0,1], skiprows=[2,3])
            tempdf = DataFrame(index = fi.index)
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
            time_ID = vidpath.split('_',1)[-1].split('.',1)[0]
            tempdf = merge_jvision_data(tempdf.reset_index(), time_ID)
            tempdf.to_pickle(storage_location + '/'+ tag + '_arena.pickle')
            print ".....", tag, " processed to pickling."
    return 


JVISION_CHANNELS = ['Frame Number',' Time Stamp (S)',' AI.0',' P0.1']
JVISION_INFORMATION = ['Frame','Timestamp','Temperature','Optostim']

def merge_jvision_data(matebookdf, searchterm):
    matebookdf['index'] = matebookdf['index'] +1   #matebook and jvision have different first frame numbers.
    filename = find_matching_jvision_metadata(searchterm)
    df = pd.read_csv(filename, sep=',')
    jdata = df[JVISION_CHANNELS]
    jdata.columns = [JVISION_INFORMATION]
    jdata['Temperature'] = (jdata['Temperature']-1.235) / 0.005
    mgd = pd.merge(matebookdf, jdata, left_on='index', right_on='Frame', how='inner')
    return mgd

def find_matching_jvision_metadata(searchterm):
    return RAWDATA_DIR + 'Metadata_' + searchterm + '.txt'
    
    
def convert_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df
    
def bin_data(df, bin_size, _HOW):
    binned = df.resample(bin_size, how=_HOW)  
    return binned


def make_rainbow_subplot(mean, sem, n, _ax, i, x_param):
        error_config = {'ecolor': '0.1'}
        
        means = mean[i]
        sems = sem[i]
        #ns = n[i]
        xs = mean[x_param]
        x = list(xs)
        y = list(means)
        psems = list(sems)
        nsems = list(-1*(sems))
        top_errbar = tuple(map(sum, zip(psems, y)))
        bottom_errbar = tuple(map(sum, zip(nsems, y)))
        p = colorline(x, y)# linewidth=2, zorder=100,color='b',) 
        q = plt.fill_between(x, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.05, 
                            zorder=90,
                            color='b',
                            )
       
        #ax.set_xlim((np.amin(x),np.amax(x)))
        if i == 'Temperature':
            _ax.set_ylim(0.9*(means.values.min()),1.1*(means.values.max()))
            _ax.set_ylabel('Temperature ('+ u"\u00b0" + 'C)', fontsize=10)
        elif i == 'DegreeSeconds':
            _ax.set_ylim(0.9*(means.values.min()),1.1*(means.values.max()))
            _ax.set_ylabel('Acc. Temp. above ' + str(THRESHOLD) + u"\u00b0" + 'C  (' + u"\u00b0" + 'C sec )', fontsize=10)
        else:    
            _ax.set_ylim(0,1.3*(means.values.max()))
            _ax.set_ylabel(INDEX_NAME[params.index(i)] + ' ' + u"\u00B1" + ' SEM', fontsize=10)   # +/- sign is u"\u00B1"
            
        laser_1 = collections.BrokenBarHCollection.span_where(x, ymin=0, ymax=1.5*(means.values.max()), where=mean['Optostim'] > 0.001, facecolor='#FFB2B2', linewidth=0, edgecolor=None, alpha=1.0, zorder=10) #green b2ffb2
        _ax.add_collection(laser_1)
        if x_param == 'Time':
            _ax.set_xlabel('Time (s)', fontsize=10)
        if x_param == 'Temperature':
            _ax.set_xlabel('Temperature ('+ u"\u00b0" + 'C)', fontsize=10)
        if x_param == 'DegreeSeconds':
            _ax.set_xlabel('Acc. Temp. above ' + str(THRESHOLD) +u"\u00b0" + 'C  (' + u"\u00b0" + 'C sec )', fontsize=10)
            _ax.set_xlim(0, mean.DegreeSeconds.max())
        _ax.tick_params(axis='both', which='major', labelsize=10)
        return
    
def plot_behaviours_per_video(mean, sem, n, exp_id, storage_location, x_param): 
    print 'plotting: ', exp_id, x_param
    fig = plt.figure(figsize=(8.5,11), dpi=300, facecolor='w')
    fig.suptitle('Behaviour vs ' + x_param + ': ' + exp_id.split('_',1)[0] + '\n' + exp_id.split('_',1)[-1] + "  -  n= " + str(n), fontsize=16)
    
    #generate temperature vs time
    axTemp = fig.add_subplot((len(params)+2)/2, 2, 1)
    if x_param == 'DegreeSeconds':
        make_rainbow_subplot(mean, sem, n, axTemp, 'DegreeSeconds', 'Time')
    else:
        make_rainbow_subplot(mean, sem, n, axTemp, 'Temperature', 'Time')
    #generate behaviour vs time or temperature
    for i in params:
        ax = fig.add_subplot((len(params)+2)/2, 2, 2+params.index(i))
        make_rainbow_subplot(mean, sem, n, ax, i, x_param)
        
        
    #l = pl.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=pl.gcf().transFigure)
    plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.87, wspace=0.40, hspace=0.32)
    #plt.show()
    fig.savefig(storage_location + '/SVG/' + exp_id + '_' + x_param + '_' + binsize_str + '.svg', bbox_inches='tight')
    fig.savefig(storage_location + '/PDF/' + exp_id + '_' + x_param + '_' + binsize_str + '.pdf', bbox_inches='tight')
    fig.savefig(storage_location + '/PNG/' + exp_id + '_' + x_param + '_' + binsize_str + '.png', bbox_inches='tight')
    plt.close(fig) 


 
    
    
def compile_data(pickle_jar):
    print 'compiling...'
    rawfile = DataFrame({'Time':[]})
    dflist = []
    vidlist = []
    flyIDlist = []
    for x in glob.glob(pickle_jar + '/*arena.pickle'):
        vidname, flyID = parse_tempdf_name(x)
        tempdf = pd.read_pickle(x)
        dflist.append(tempdf)
        vidlist.append(vidname)
        flyIDlist.append(flyID)
    rawfile = pd.concat(dflist, keys=flyIDlist, names=['Arena'])
    rawfile = rawfile.reset_index()
    rawfile = convert_timestamps(rawfile)
    #rawfile.to_csv(OUTPUT + 'rawfile.csv', sep=',')
    rawfile.to_pickle(pickle_jar + '/' + vidname +  '_compiled.pickle')
    return rawfile



def combine_annotations(rawfile, filefile):
    print "merging with fly info..."
    for x in filefile.columns:
        rawfile[x] = filefile[x].reindex(rawfile.index, level=0)
    rawfile.to_pickle(JAR + 'df.pickle')
    rawfile.to_csv(OUTPUT + 'df.csv', sep=',')
    return

def group_data(rawfile):
    
    mean = rawfile.resample(binsize_str, how='mean')
    std = rawfile.resample(binsize_str, how='std')
    n = len(set(rawfile.Arena))
    sem = std / np.sqrt(n)
    
    return mean, sem, n
    
def generate_cumulative_temperature(mean, threshold):
    mean['degrees_above_threshold'] = mean['Temperature'] - threshold
    mean['degrees_above_threshold'][mean['degrees_above_threshold'] <= 0.0] = 0.0
    mean['DegreeSeconds'] = mean.degrees_above_threshold.cumsum()*binsize
    return mean


############################################multi-colour tools   ########################################################################

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, zorder=100):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 



############################################################################################################

experiment_list=[]
for x in glob.glob(INPUT_DIR + '*.mbr'):
    
    exp_id = parse_screen_filename(x)
    print exp_id
    if exp_id in experiment_list:
        continue
    else:
        experiment_list.append(exp_id)
        
        
   
        
for experiment in experiment_list:
    if not os.path.exists(OUTPUT_DIR + '/SUMMARY_FILES/PNG/' + experiment+ '**' + binsize_str + '.png'):
        storage_location = OUTPUT_DIR + 'JAR/' + experiment
        if not os.path.exists(storage_location):
            os.makedirs(storage_location)
        if plots_only == False:
            for directory in glob.glob(INPUT_DIR + experiment + '*'):
                exp_id, timestamp = parse_tempdf_name(directory)
                process_matebook_data(directory, _paramlist, storage_location)
        
        compiled = compile_data(storage_location)
        _mean, _sem, _n = group_data(compiled)
        _mean = generate_cumulative_temperature(_mean, THRESHOLD)
        _sem = generate_cumulative_temperature(_sem, THRESHOLD)
        
        plot_behaviours_per_video(_mean, _sem, _n, experiment, OUTPUT_DIR + '/SUMMARY_FILES/', 'Time')
        plot_behaviours_per_video(_mean, _sem, _n, experiment, OUTPUT_DIR + '/SUMMARY_FILES/', 'Temperature')
        plot_behaviours_per_video(_mean, _sem, _n, experiment, OUTPUT_DIR + '/SUMMARY_FILES/', 'DegreeSeconds')
 
utilities.sendMail(args.email, 'Matebook processing finished', ('Your files are available at: ' + OUTPUT_DIR))
print "Done."
