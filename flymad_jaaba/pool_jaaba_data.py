import os, fnmatch
import shutil
import os
import argparse
import pandas as pd
from pandas import DataFrame
import glob
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--searchdir', type=str, required=True,
                        help='directory of original matebook files')  
parser.add_argument('--searchterm', type=str, required=True,
                        help='match filename with string, ex *uas*')
parser.add_argument('--pooldir', type=str, required=True,
                        help='directory to copy files for pooling')

args = parser.parse_args()

SEARCH_TERM = args.searchterm
SEARCH_DIR = args.searchdir
POOL_DIR = args.pooldir
if not os.path.exists(POOL_DIR) ==True:
    os.makedirs(POOL_DIR )
if not os.path.exists(POOL_DIR + 'BAGS/') ==True:
    os.makedirs(POOL_DIR + 'BAGS/' )
filelist = []

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def parse_filename(path):
    vidname = path.split('.mbr/')[1] + '.mbr'
    vidname = vidname.rsplit('.mbr')[0] + '.mbr'
    flyID = path.split('.mbr/')[1]
    flyID = flyID.split('/track.tsv',1)[0]
    return vidname, flyID

def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def parse_bagtime(namestring):
    numstr = namestring.split('/')[-1].split('_')[-1].split('.bag')[0].replace('-','')
    bagtime = pd.to_datetime(numstr)
    return bagtime
    
def match_fmf_and_bag(fmftime):
    #print 'from mat_fmf_and_bag: ', fmftime
    fmftime64 = np.datetime64(fmftime)
    bagtime = bagframe['Timestamp'].asof(fmftime)
    #print "bagtime: ", bagtime
    if fmftime64 - bagtime > np.timedelta64(30000000000, 'ns'):
        print "ERROR: fmf is more than 30 seconds younger than bagfile: ", fmftime
    bagfile = bagframe['Filepath'].asof(fmftime)
    return bagfile


"""
def pick_matching_dirs(directory, pattern1, pattern2):   #example: '/groups/dickson/dickson..../.mbd', '*uas*', 'track.tsv'
    for x in glob.glob(directory + '/' + pattern1):   #*uas*, for example
        fullpath = find_files(x, pattern2)   #track.tsv, for example
    return fullpath
"""
    
def copy_with_dirs(fullpath):
    newpath = POOL_DIR + '/'.join(fullpath.split('/')[-2:])
    if not os.path.exists(newpath) ==True:
        os.makedirs('/'.join(newpath.rsplit('/',2)[0]) + '/')
    shutil.copy(fullpath, newpath)
    

def copy_bag_file(fullpath):
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(fullpath.split('/registered_trx.csv')[0])
    BAG_FILE = match_fmf_and_bag(FMF_TIME)
    shutil.copy(BAG_FILE, (POOL_DIR + 'BAGS/'))

baglist = []
for bag in glob.glob(SEARCH_DIR + 'BAGS/*.bag'):
    bagtimestamp = parse_bagtime(bag)
    baglist.append((bag, bagtimestamp))
bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
bagframe.index = pd.to_datetime(bagframe['Timestamp'])
bagframe = bagframe.sort()

print SEARCH_DIR + SEARCH_TERM
    
for matching_dir in glob.glob(SEARCH_DIR + SEARCH_TERM):
    filelist.append(matching_dir)
    print matching_dir
    for fn in find_files(matching_dir, 'frame_by_frame_synced.pickle'):
        print fn
        copy_with_dirs(fn)
        print fn.rsplit('/',1)[0] + '/tracking_info.pickle'
        copy_with_dirs(fn.rsplit('/',1)[0] + '/tracking_info.pickle')
        copy_with_dirs(fn.rsplit('/',1)[0] + '/wingdata.pickle')
        copy_bag_file(fn)



fileDF = DataFrame(filelist)
fileDF.to_csv(POOL_DIR + 'filelist.txt', sep='\n', header=None, index=None)
