

import os, fnmatch
import numpy as np
import pandas as pd
from pandas import DataFrame
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='directory of Matebook Project files')  
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to store analysis')
parser.add_argument('--filedir', type=str, required=False,
                        help='directory of list of genotype information')
parser.add_argument('--fps', type=float, required = True, 
                        help='framerate in FPS, ex: 25.0')
args = parser.parse_args()

DROP = args.inputdir

OUTPUT = args.outputdir

FPS = args.fps






def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

files = []
times = []

for filename in find_files(DROP, 'track.tsv'):
    fi = pd.read_table(filename, sep='\t', skiprows=[0,2,3])
    copstart = np.argmax(fi['copulating'])
    if len(set(fi['copulating'])) <= 1:
        copstart = np.nan
        print '\t', filename, ' did not copulate'
    else:
        pass 
    files.append(filename)
    times.append(copstart/FPS)
    print 'appending:', filename, '  ', copstart
        
    
df = DataFrame(zip(files, times), columns=['Fly ID','Latency to copulation (s)'])

df.to_csv(OUTPUT + 'latency_to_copulation.csv', sep=',')






