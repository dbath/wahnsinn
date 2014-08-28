

import os, fnmatch
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl
from scipy import stats as st


DROP = '/tier2/dickson/DN_screen'   #Location of behavior.tsv and filenames.csv files



def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                
                yield filename

files = []
times = []

for filename in find_files(DROP, '*.MTS'):
    if 'DN_scr_96' in filename:
                    print filename






