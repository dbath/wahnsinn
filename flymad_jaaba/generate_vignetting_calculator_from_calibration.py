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
newfmf = FMF.FlyMovieSaver(sys.argv[2] + '.fmf')

threshold = float(sys.argv[3]) #  10
print threshold
proportion_dark = float(sys.argv[4])  #  0.0003
print proportion_dark
listy = []
listynew = []
times = []
timesnew = []
sumframe = np.zeros((fmf.get_width(),fmf.get_height()))
for frame_number in range(fmf.get_n_frames()):
    frame,timestamp = fmf.get_frame(frame_number)
    zeros = float(frame[frame <= threshold].size)
    nonzeros = float(frame[frame >= threshold].size)
    listy.append(zeros/nonzeros)
    times.append(frame_number)    
    if (zeros/nonzeros) <= proportion_dark:
        newfmf.add_frame(frame, timestamp)
        sumframe = sumframe + frame
        listynew.append(zeros/nonzeros)
        timesnew.append(frame_number)
    else:
        pass

    
newfmf.close()


#plot proportion of black pixels before and after
plt.plot(times, listy, color='b')
plt.plot(timesnew, listynew, color='r')
plt.save_fig(sys.argv[2] + '_zeros.svg')
plt.show()

#plot average intensity (vignette)
vignette = sumframe / len(timesnew)

plt.imshow(vignette)
plt.save_fig(sys.argv[2] + '_vignette.svg')
plt.show()

#Save vignette values
DataFrame(vignette).to_csv(sys.argv[2] + '_array.csv', sep=',')

print 'done.'
