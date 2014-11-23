import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import matplotlib.cm as cm
from pytz import common_timezones
import rosbag
import argparse
import glob
from scipy import stats as st
import time
import os.path
import glob
import tempfile
import shutil
import re
import collections
import operator
import multiprocessing
import sys, os
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import benu.benu
import benu.utils
from PIL import Image, ImageStat

def make_annotated_mp4(fmf, bag, trx):  # fmf movie, rosbag file, and csv output from jaaba tracking
    
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
    fmftime64 = np.datetime64(fmftime)
    bagtime = bagframe['Timestamp'].asof(fmftime)
    if fmftime64 - bagtime > np.timedelta64(30000000000, 'ns'):
        print "ERROR: fmf is more than 30 seconds younger than bagfile: ", fmftime
    bagfile = bagframe['Filepath'].asof(fmftime)
    return bagfile    
    
#MAIN


def process_directory_of_files(directory):
    for fmf_file in glob.glob(directory + '/*.fmf'):
        fmf = FMF.FlyMovie(fmf_file)
        flyID, fmf_file_time, expID = parse_fmftime(fmf_file.rsplit('.')[-2])
        for bag in glob.glob(directory + '/BAGS/*.bag'):
            bagtime = parse_bagtime(bag)
            if abs(bagtime - 
        
