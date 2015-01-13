import os.path
import os
import glob
import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import pandas as pd
from pandas import DataFrame
import numpy as np
import argparse
import cv2
import shutil
import flymad_jaaba.utilities as utilities



class TargetDetector(object):
    """
    when passed an FMF object or path to an FMF file, this class generates a background image and detects objects within it.
    """
    def __init__(self, fmf_filepath, tempdir=None, sample_rate=500):
        """
        
        """
        self._fmf_filepath = fmf_filepath
        
        if tempdir is not None:
            if tempdir[-1] == '/':
                pass
            else:
                tempdir = tempdir + '/'
            self._tempdir = tempdir
        else:
            self._tempdir = fmf_filepath.split('.fmf')[0] + '/targets/'
        
        if not os.path.exists(self._tempdir) == True:
            os.makedirs(self._tempdir)
        
        self._sample_rate = sample_rate    # = n for every nth frame
        """
        self._targets = self.detect_targets(self._fmf_filepath, self._tempdir)
        
        if tempdir == None:
            shutil.rmtree(self._tempdir)
        else:
            self.plot_targets_on_background((tempdir+'background.png'), self._targets, self._tempdir)
        """
        return
    
        
            
        
        
    def create_png(self, image_array):
        self._image_height, self._image_width = image_array.shape
        fig = plt.figure(frameon=False)
        fig.set_size_inches(self._image_height, self._image_width)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_array, cmap = cm.Greys_r)
        return fig 
        
        
    def generate_background_image(self):
    
        wide_fmf = FMF.FlyMovie(self._fmf_filepath)
        frame0, _ = wide_fmf.get_frame(0)
        image_width, image_height = frame0.shape
        acc=np.zeros((image_width, image_height), np.float32) # 32 bit accumulator
        for frame_number in range(0, wide_fmf.get_n_frames(), self._sample_rate):
            frame, timestamp = wide_fmf.get_frame(frame_number)
            acc = np.maximum.reduce([frame, acc])
        self.create_png(acc)
        plt.savefig(self._tempdir + 'background.png', dpi=1)
        plt.close('all')
        return
        
    def get_dirs(self):
        return self._fmf_filepath, self._tempdir
             
    def detect_targets(self):#, _fmf_filepath, _tempdir):
    
        if not os.path.exists(self._tempdir + 'background.png'):
            self.generate_background_image()
        
        im = cv2.imread(self._tempdir + 'background.png')
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
        #BINARIZE AND DETECT CONTOURS FOR TARGETS
        ret, thresh = cv2.threshold(imgray, 128, 255, 0)
        kernel = np.ones((5,5),np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
    
        contours_targets, hierarchy = cv2.findContours(eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)    
        self._targets = []
        for c in contours_targets:
            area = cv2.contourArea(c)
            if (area >= 50) and (area <=300):
                (cx,cy), radius = cv2.minEnclosingCircle(c)
                self._targets.append([cx,cy, radius])
        return self._targets

    def plot_targets_on_background(self):
        targets = self.detect_targets()
        background_image = cv2.imread(self._tempdir + 'background.png')
        for coords in targets:
            x = int(np.rint(coords[0]))
            y = int(np.rint(coords[1]))
            r = int(np.rint(coords[2]))
            cv2.circle(background_image, (x, y), r, (255, 0, 0), 2)
        cv2.imshow('targets', background_image)
        #cv2.waitKey(0)
        cv2.imwrite((self._tempdir + 'targets.png'), background_image)
        cv2.destroyAllWindows()  
    
    def get_targets(self):
        return self._targets
    
    def get_dist_to_nearest_target(self, bagfile):
        
        self._targets = self.detect_targets()
            
        positions = utilities.get_positions_from_bag(bagfile)
        
        distances = DataFrame()
        for target in range(len(self._targets)):
            px, py, pr = self._targets[target]
            distances['d'+ str(target)] = (((positions['fly_x'] - px)**2 + (positions['fly_y'] - py)**2)**0.5) - pr
        
        distances['Timestamp'] = positions.Timestamp
        distances = utilities.convert_timestamps(distances)
        
        dtarget = DataFrame(distances.min(axis=1), columns=['dtarget'])
        return dtarget

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--widefmf', type=str, required=True,
                        help='path to fmf') 

    args = parser.parse_args()

    targets = TargetDetector(args.widefmf)
    print targets.detect_targets()
    
    
"""
USAGE:

run target_detector.py with args

or from another file, example:

foo = target_detector.TargetDetector('/media/DBATH_6/150108/DB201-GP-costim_wide_20150108_112916.fmf')

first_target_coords = foo.detect_targets()[0]

x_coords, y_coords, radius = foo.detect_targets()[0]

"""

