import time
import os.path
import glob
import tempfile
import shutil
import re
import collections
import operator
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import pandas as pd
import numpy as np

import benu.benu
import benu.utils

from PIL import Image, ImageStat

#import roslib; roslib.load_manifest('flymad')
#import flymad.madplot as madplot

fname = sys.argv[1]
fmf = FMF.FlyMovie(fname)
newfmf = FMF.FlyMovieSaver(sys.argv[2])
listy = []
listynew = []
times = []
timesnew = []
for frame_number in range(fmf.get_n_frames()):
    frame,timestamp = fmf.get_frame(frame_number)
    zeros = float(frame[frame <= 10].size)
    nonzeros = float(frame[frame >= 10].size)
    if zeros/nonzeros <= 0.0003:
        newfmf.add_frame(frame, timestamp)
        listynew.append(zeros/nonzeros)
        timesnew.append(frame_number)
    else:
        pass
    listy.append(zeros/nonzeros)
    times.append(frame_number)
newfmf.close()

plt.plot(times, listy, color='b')
plt.plot(timesnew, listynew, color='r')
plt.show()

print 'done.'
