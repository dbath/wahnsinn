import os, fnmatch
import shutil
import os
import argparse
import pandas as pd
from pandas import DataFrame
import glob


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--searchdir', type=str, required=True,
                        help='directory of registered_trx.csv files')  
parser.add_argument('--searchterm', type=str, required=True,
                        help='match filename with string, ex *uas*')
parser.add_argument('--pooldir', type=str, required=True,
                        help='directory to copy files for pooling')

args = parser.parse_args()

SEARCH_TERM = args.searchterm
SEARCH_DIR = args.searchdir
if (SEARCH_DIR[-1] != '/'):
    SEARCH_DIR = SEARCH_DIR + '/'
POOL_DIR = args.pooldir
if (POOL_DIR[-1] != '/'):
    POOL_DIR = POOL_DIR + '/'

filelist = []

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def copy_with_dirs(fullpath):
    newpath = POOL_DIR + '/'.join(fullpath.split('/')[-2:])
    if not os.path.exists('/'.join(newpath.split('/')[:-1]) + '/') ==True:
        os.makedirs('/'.join(newpath.split('/')[:-1]) + '/')
    shutil.copy(fullpath, newpath)
    
    
for matching_dir in glob.glob(SEARCH_DIR + SEARCH_TERM):
    filelist.append(matching_dir)
    for fn in find_files(matching_dir, 'wingdata.pickle'):
        copy_with_dirs(fn)
    for fn in find_files(matching_dir, 'frame_by_frame_synced.pickle'):
        copy_with_dirs(fn)
    for fn in find_files(matching_dir, 'tracking_info.pickle'):
        copy_with_dirs(fn)
    



fileDF = DataFrame(filelist)
fileDF.to_csv(POOL_DIR + 'filelist.txt', sep='\n', header=None, index=None)
