import glob
import shutil
import os
import argparse
import pickle
import pandas as pd
from pandas import DataFrame

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='location of fmfs')  
args = parser.parse_args()

INPUTDIR = args.inputdir
filelist = ['experiment_name', INPUTDIR + 'params.xml']
for fmf in glob.glob(INPUTDIR + '*.fmf'):
    print fmf
    if 'zoom' in fmf:
        newdir = fmf.split('.fmf')[0]
        filelist.append(newdir)
        os.makedirs(newdir)
        shutil.move(fmf, newdir+'/')
        shutil.copy('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/JAABA_BS_files/Metadata.xml', newdir+'/') #silly.
        shutil.copy('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/JAABA_BS_files/params.xml', newdir+'/') #silly.
fileDF = DataFrame(filelist)
fileDF.to_csv(INPUTDIR + 'filelist.txt', sep='\n', header=None, index=None)

