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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
newfmf = FMF.FlyMovieSaver(sys.argv[2] + 'newfmf.fmf')


starting_timestamp = fmf.get_frame(0)[1]


def CreateMovie(plotter, numberOfFrames, fps=10):
 
	for i in range(numberOfFrames):
		plotter(i)

		fname = '_tmp%05d.png'%i
 
		plt.savefig(fname)
		plt.clf()
 
	os.system("rm movie.mp4")

	os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie.mp4")
	os.system("rm _tmp*.png")

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fmf2fig(frame, timestamp, colourmap_choice):
    #elapsed_time = int(timestamp - starting_timestamp)
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    ax.imshow(frame, cmap = colourmap_choice)

    ax.text(0.95, 0.05, pd.to_datetime(timestamp, unit='s'), 
            verticalalignment='top', horizontalalignment='left', 
            color='white', fontsize=10)
    
    return figure

def add_wingAngle(frame, synced_time, Left, Right):
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    ax.imshow(frame)
    ax.text(0.95, 0.01, pd.to_datetime(synced_time, unit='s'), verticalalignment='top', horizontalalignment='left', color='white', fontsize=14)
    

for frame_number in range(fmf.get_n_frames()):
    frame,timestamp = fmf.get_frame(frame_number)
    figure = fmf2fig(frame, timestamp, cm.Greys_r)
    #newframe = fig2data(figure)
    #newfmf.add_frame(newframe, timestamp)
    plt.savefig(sys.argv[2] + '_tmp%05d.png'#%frame_number)
    plt.clf()
    plt.close()
    
    
newfmf.close()

print 'done.'

"""

to make an mp4 from the png output of this script:

move to directory where pngs are stored, and:

ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -b 5000k -y SS01538-UC-RIR_zoom_141105_094600.mp4

where:
-f  image2      specifies that it is a series of images
-r 15           specifies 15 FPS
-i _tmp%05d.png specifies file name pattern. silly bug means that first file must be less than 00004.
-vcodec mpeg4   specifies make mp4
-b 5000k        specifies bitrate (500kbps?)
-y              not sure
last part is output filename.




