import os.path
import os
import glob
import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import pandas as pd
from pandas import DataFrame
import numpy as np
import argparse
import cv2
import shutil
import flymad_jaaba.utilities as utilities
import roslib; roslib.load_manifest('flymad')
import flymad.madplot as madplot
import flymad.flymad_analysis_dan as flymad_analysis



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
        
        self._calibration_file = utilities.get_calibration_asof_filename(self._fmf_filepath)
        
        self._arena = madplot.Arena(convert=False, **flymad_analysis.get_arena_conf(calibration_file=self._calibration_file))
        
        self._sample_rate = sample_rate    # = n for every nth frame
        
        self._targets = self.detect_targets()
        
        
    
        
            
        
        
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
        image_height, image_width = frame0.shape
        acc=np.zeros((image_height, image_width), np.float32) # 32 bit accumulator
        for frame_number in range(0, wide_fmf.get_n_frames(), self._sample_rate):
            frame, timestamp = wide_fmf.get_frame(frame_number)
            acc = np.maximum.reduce([frame, acc])
        cv2.imshow('background',acc)
        cv2.imwrite((self._tempdir + 'background.png'), acc)
        cv2.destroyAllWindows()
        return
        
    def get_dirs(self):
        return self._fmf_filepath, self._tempdir
             
    def detect_targets(self):#, _fmf_filepath, _tempdir):
        
        (cx, cy), cr = self._arena.circ
    
        if not os.path.exists(self._tempdir + 'background.png'):
            self.generate_background_image()
        
        im = cv2.imread(self._tempdir + 'background.png')
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
        #BINARIZE AND DETECT CONTOURS FOR TARGETS
        ret, thresh = cv2.threshold(imgray, 128, 255, 0)
        kernel = np.ones((3,3),np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
    
        contours_targets, hierarchy = cv2.findContours(eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)    
        self._targets = []
        for c in contours_targets:
            area = cv2.contourArea(c)
            #TARGETS MUST BE APPROPRIATE SIZE
            if (area >= 10) and (area <=400):
                (x,y), radius = cv2.minEnclosingCircle(c)
                #CHECK IF TARGET IN CENTRE 40% OF ARENA
                if ( ((x - cx)**2 + (y - cy)**2) <= (cr*0.4)**2): 
                    self._targets.append([x,y, radius])
        return self._targets

    def plot_targets_on_background(self):
        targets = self.detect_targets()
        background_image = cv2.imread(self._tempdir + 'background.png')
        for coords in targets:
            x = int(np.rint(coords[0]))
            y = int(np.rint(coords[1]))
            r = int(np.rint(coords[2]))
            cv2.circle(background_image, (x, y), r, (255, 0, 0), 2)
        
        (cx, cy), cr = self._arena.circ
        cv2.circle(background_image, (int(cx), int(cy)),int(cr), (0,255,0),2)
        #cv2.circle(background_image, (int(cx), int(cy)),int(cr*.6), (0,128,0),1)
        cv2.imshow('targets', background_image)
        
        #cv2.waitKey(0)
        cv2.imwrite((self._tempdir + 'targets.png'), background_image)
        cv2.destroyAllWindows()  
    
    def plot_trajectory_on_background(self, bagfile):
        
        if not os.path.exists(self._tempdir + 'targets.png'):
            self.plot_targets_on_background()
        
        im = cv2.imread(self._tempdir + 'targets.png')
        positions = utilities.get_positions_from_bag(bagfile)
        distances = self.get_dist_to_nearest_target(bagfile)
        for ts in positions.index:
            x = int(np.rint(positions.fly_x[ts]))
            y = int(np.rint(positions.fly_y[ts]))
            d = int(np.rint(distances.dtarget[ts]))
            
            fraction = (float(ts)/len(positions))
            if fraction <= 0.5:
                red = 255.0*2*(fraction)
                green = 255.0*2*(fraction)
                blue = 255.0
            if fraction > 0.5:
                red = 255.0
                green = 255.0 - 255.0*2*(fraction - 0.5)
                blue = 255.0 - 255.0*2*(fraction - 0.5)
            
            cv2.circle(im, (x,y), 1, (blue, green, red), -1)
        cv2.imshow('traj', im)
        cv2.imwrite((self._tempdir + 'trajectory.png'), im)
        cv2.destroyAllWindows() 
    
    def plot_trajectory_and_wingext(self, WING_DF, BAG_FILE, TS_START=None, TS_END=None, FN='full', background=True):#, TS_START, TS_END):
        """ Plots trajectory with wing angle indicated in colourmap. 
            Give start and end timestamps to limit trajectory to a section of the experiment.
        """
        if TS_START is None:
            TS_START = WING_DF.index[0]
        if TS_END is None:
            TS_END = WING_DF.index[-1]
        
        WING_DF = WING_DF[TS_START:TS_END]
        WING_DF = WING_DF.sort('maxWingAngle')
        print FN, TS_END-TS_START
 
        im = cv2.imread(self._tempdir + 'background.png')
       
        if background == True:
            im = cv2.imread(self._tempdir + 'background.png')
            fig = gcf()
            ax = fig.gca(frameon=False) 
            s = ax.scatter(WING_DF.fly_x,WING_DF.fly_y, c=WING_DF.maxWingAngle,vmin=0.0, vmax=2.1, zorder=1, s=7, linewidths=0   ) 
        elif background == False:
            img = cv2.imread(self._tempdir + 'background.png')
            im = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8) +255
            fig = gcf()
            ax = fig.gca(frameon=False)
            targets = self.detect_targets()
            for coords in targets:
                x = int(np.rint(coords[0]))
                y = int(np.rint(coords[1]))
                r = int(np.rint(coords[2]))
                cv2.circle(im, (x, y), r-2, (100, 100, 100), -1)
        
            (cx, cy), cr = self._arena.circ
            cv2.circle(im, (int(cx), int(cy)),int(cr)-8, (20, 20, 20),2)
            s = ax.scatter(WING_DF.fly_x,WING_DF.fly_y, c=WING_DF.maxWingAngle,vmin=0.0, vmax=2.1, zorder=1, s=17, linewidths=0   ) 
            

          
        plt.imshow(im)#, zorder=0)   
        plt.axis('off')
        cbaxes = fig.add_axes([0.9,0.2,0.025,0.60])
        cbar = plt.colorbar(s, cax = cbaxes )
        cbar.set_label("Wing Extension Angle (rad)")
        cbar.ax.tick_params(labelsize=10)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.savefig(self._tempdir + 'wingext_trajectory_' + FN + '.svg', bbox_inches='tight', pad_inches=0.0   )
        plt.savefig(self._tempdir + 'wingext_trajectory_' + FN + '.pdf', bbox_inches='tight', pad_inches=0.0   )
        plt.savefig(self._tempdir + 'wingext_trajectory_' + FN + '.png', bbox_inches='tight', pad_inches=0.0   )
        plt.close('all')

    def get_targets(self):
        return self._targets
    
    def get_dist_to_nearest_target(self, bagfile):
        
        self._targets = self.detect_targets()
            
        positions = utilities.get_positions_from_bag(bagfile)
        
        distances = DataFrame()
        for target in range(len(self._targets)):
            px, py, pr = self._targets[target]
            distances['d'+ str(target)] = ((((positions['fly_x'] - px)/5.2)**2 + ((positions['fly_y'] - py)/4.8)**2)**0.5)# - pr
        
        distances['Timestamp'] = positions.Timestamp
        distances = utilities.convert_timestamps(distances)
        
        self.dtarget = DataFrame(distances.min(axis=1), columns=['dtarget'])
        return self.dtarget

    def get_dtarget_at_timestamp(self, _timestamp):
        data = self.dtarget[self.dtarget.Timestamp >= _timestamp].iloc[0]
        return data['dtarget'].values
        
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

