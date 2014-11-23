import os, fnmatch
import shutil
import os
import argparse
import pandas as pd
from pandas import DataFrame
import glob


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

"""
def pick_matching_dirs(directory, pattern1, pattern2):   #example: '/groups/dickson/dickson..../.mbd', '*uas*', 'track.tsv'
    for x in glob.glob(directory + '/' + pattern1):   #*uas*, for example
        fullpath = find_files(x, pattern2)   #track.tsv, for example
    return fullpath
"""
    
def copy_with_dirs(fullpath):
    newpath = POOL_DIR + '/'.join(fullpath.split('/')[-3:])
    if not os.path.exists(newpath) ==True:
        os.makedirs('/'.join(newpath.split('/')[:-1]) + '/')
    shutil.copy(fullpath, newpath)
    
    
for matching_dir in glob.glob(SEARCH_DIR + SEARCH_TERM):
    filelist.append(matching_dir)
    for fn in find_files(matching_dir, 'track.tsv'):
        copy_with_dirs(fn)
    



fileDF = DataFrame(filelist)
fileDF.to_csv(POOL_DIR + 'filelist.txt', sep='\n', header=None, index=None)
