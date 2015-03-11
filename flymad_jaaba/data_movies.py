import time
import os
import glob
import tempfile
import datetime
import shutil
import re
import matplotlib.collections as collections
import operator
import multiprocessing
import sys, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import pandas as pd
from pandas import DataFrame
import numpy as np
import rosbag
import rosbag_pandas
import benu.benu
import benu.utils
from PIL import Image, ImageStat
import argparse
import flymad_jaaba.utilities as utilities
import flymad_jaaba.target_detector as target_detector
import flymad_jaaba_v4 as flymad_jaaba_v4


class FlyPanel(object):
    """
    pass an fmf, data, and panel position to generate subplots.
    """
    def __init__(self, fmf_dir, savedir, fly_id, plot_overlays=False):
        
        if (savedir[-1] == '/' ):
            savedir = savedir[:-1]
        
        if not os.path.exists(savedir + '/temp_png'):
            os.makedirs(savedir + '/temp_png')

        self._savedir = savedir

        if (fmf_dir[-1] == '/'):
            self._fmf_dir  = fmf_dir[:-1]
        else:
            self._fmf_dir  = fmf_dir
            
        self._plot_overlays = plot_overlays
        
        self._expdir = fmf_dir.rsplit('/', 1)[0]
        
        
        self._bag = utilities.match_fmf_and_bag(self._fmf_dir, (self._expdir + '/BAGS'))
        
        self._wide = utilities.match_wide_to_zoom(self._fmf_dir, (self._expdir + '/'))
        
        self._handle, __, ___ = utilities.parse_fmftime(self._fmf_dir)
        
        self._zoomfmf, self._data, self._Tzero = self.get_data()
        
        self._image_height, self._image_width = self._zoomfmf.get_frame(0)[0].shape
        
        

    def get_frame(self, framenumber):
        return self._zoomfmf.get_frame(framenumber)
        
    def get_n_frames(self):
        return self._zoomfmf.get_n_frames()
           
    def sync_jaaba_with_ros(self, FMF_DIR, BAGS, JAABA):

        print "Processing: ", FMF_DIR
        
        JAABA_CSV               = FMF_DIR + '/registered_trx.csv'
        
        FLY_ID, FMF_TIME, GROUP = utilities.parse_fmftime(FMF_DIR)
        
        BAG_FILE                = utilities.match_fmf_and_bag(FMF_DIR, BAGS)
        
        WIDE_FMF                = utilities.match_wide_to_zoom(FMF_DIR, JAABA)
        
        jaaba_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Length','Width','Theta','Left','Right'], index_col=False)
        jaaba_data[['Length','Width','Left','Right']] = jaaba_data[['Length','Width','Left','Right']].astype(np.float64)
        jaaba_data = utilities.convert_timestamps(jaaba_data)
        
        # ALIGN LASER STATE DATA
        jaaba_data['Laser1_state'] = utilities.binarize_laser_data(BAG_FILE, 2).resample('67ms', how='max')['Laser_state'].asof(jaaba_data.index).fillna(value=0)  #YAY! 
        jaaba_data['Laser2_state'] = utilities.binarize_laser_data(BAG_FILE, 4).resample('67ms', how='max')['Laser_state'].asof(jaaba_data.index).fillna(value=0)
        
        # COMPUTE AND ALIGN DISTANCE TO NEAREST TARGET
        targets = target_detector.TargetDetector(WIDE_FMF, FMF_DIR)
        targets.plot_targets_on_background()
        jaaba_data['dtarget'] = targets.get_dist_to_nearest_target(BAG_FILE)['dtarget'].asof(jaaba_data.index).fillna(value=0)
        
        jaaba_data['Timestamp'] = jaaba_data.index  #silly pandas bug for subtracting from datetimeindex...
        try:
            jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data.Timestamp[(jaaba_data.Laser2_state + jaaba_data.Laser1_state) > 0].index[0]
        except:
            print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
            jaaba_data['synced_time'] = jaaba_data['Timestamp'] - jaaba_data.Timestamp.index[0]   
        
        #DISCARD BOGUS WING ANGLE VALUES:
        jaaba_data['Left'][jaaba_data['Left'] < -2.09 ]= np.nan   #2.09 for 120degrees
        jaaba_data['Right'][jaaba_data['Right'] > 2.09] = np.nan
        
        jaaba_data['maxWingAngle'] = jaaba_data[['Left','Right']].abs().max(axis=1)
        

        return jaaba_data


    def get_data(self):
        for x in glob.glob(self._fmf_dir + '/*.fmf'):
            fmf =  FMF.FlyMovie(x)
            
        if not os.path.exists(self._savedir +'/'+ self._handle + '_cache.pickle'):
            data = self.sync_jaaba_with_ros(self._fmf_dir, (self._expdir + '/BAGS'), (self._expdir + '/'))
            data.to_pickle(self._savedir +'/'+ self._handle + '_cache.pickle')
        data = pd.read_pickle(self._savedir +'/'+ self._handle + '_cache.pickle')
        try:
            zero_timestamp = data['Timestamp'][data[data.Laser1_state + data.Laser2_state > 0].index[0]]
        except:
            print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
            zero_timestamp = data['Timestamp'].index[0]
        
        zero_ts_float = (np.datetime64(zero_timestamp.to_datetime()) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        frame_at_t0 = self.get_frame_number_at_or_before_timestamp(fmf, zero_ts_float)
        return fmf, data, frame_at_t0    
    
    def get_zero_timestamp(self):
        try:
            zero_timestamp = self._data['Timestamp'][self._data[self._data.Laser1_state + self._data.Laser2_state > 0].index[0]]
        except:
            print "WARNING:   Cannot synchronize by stimulus. Setting T0 to frame0. "
            zero_timestamp = self._data['Timestamp'].index[0]
        
        zero_ts_float = (np.datetime64(zero_timestamp.to_datetime()) - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return zero_ts_float
        
        
    
    def get_frame_number_at_or_before_timestamp(self, fmf_object, timestamp): #stolen and modified from motmot!
        tss = fmf_object.get_all_timestamps()
        at_or_before_timestamp_cond = tss <= timestamp
        nz = np.nonzero(at_or_before_timestamp_cond)
        if len(nz)==0:
            raise ValueError("no frames at or before timestamp given")
        fno = nz[0][-1]
        return fno
        
    def get_frame_at_or_before_timestamp(self, fmf_object, timestamp): #stolen and modified from motmot!
        tss = fmf_object.get_all_timestamps()
        at_or_before_timestamp_cond = tss <= timestamp
        nz = np.nonzero(at_or_before_timestamp_cond)
        if len(nz)==0:
            raise ValueError("no frames at or before timestamp given")
        fno = nz[0][0]
        return fno    
    
    
    def set_overlays(self, toggle):
        if toggle == 'on':
            self._plot_overlays = True
        if toggle == 'off':
            self._plot_overlays = False        
        else:
            self._plot_overlays = False
        return self._plot_overlays
    
    def plot_zoom(self, frame, timestamp, colourmap_choice, jaaba, ax):
        
        image_height, image_width = frame.shape
        
        frame = np.flipud(frame)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax.imshow(frame, cmap = colourmap_choice)
        ax.set_xticks([]); ax.set_yticks([])
        np.set_printoptions(precision=2)
        ax.text(0.01*(image_width), 0.01*(image_height), 
                str(np.around(jaaba.ix[0].synced_time / np.timedelta64(1,'s'), 2)) +  's', 
                verticalalignment='top', horizontalalignment='left', 
                color='white', fontsize=22)

        RED_alpha = float(jaaba.Laser2_state)
        IR_alpha = float(jaaba.Laser1_state)
        RED_ind = plt.Circle((0.9*(image_width), 0.1*(image_height)), 0.05*(image_width), color='r', alpha=RED_alpha)
        IR_ind = plt.Circle((0.75*(image_width), 0.1*(image_height)), 0.05*(image_width), color='k', alpha=IR_alpha)    
        ax.add_patch(RED_ind)
        ax.add_patch(IR_ind)        
        #ax.text(0.65*(image_width), 0.1*(image_height), 'Laser:', horizontalalignment='right', verticalalignment='top',
        #                                                          color='white', fontsize=22)
        
        if self._plot_overlays ==True:
            #plot wing angles. use 0.174 multiplier to make 2.3rad equal to 40% of image height.   
            LeftAngle = float(jaaba.Left)
            RightAngle = float(jaaba.Right)     
            left = plt.Rectangle((image_width-110,image_height/2), 50, LeftAngle*0.174*image_height, color='#FF0000')
            right = plt.Rectangle((image_width-110, image_height/2), 50, RightAngle*0.174*image_height, color='#00FF00')
            #plot proximity
            proximity_val = float((270 - jaaba.dtarget)*((0.8*image_height)/270))#with max dtarget of 270, 2.963 makes max bar height 800.
            proxi = plt.Rectangle((image_width-50,((image_height/2) - (proximity_val/2))), 50, proximity_val, color='#0000FF', alpha=1.0)
            ax.add_patch(left)
            ax.add_patch(right)
            ax.add_patch(proxi)
            ax.text(image_width-85, (image_height - 20)/2, 'L', 
                    horizontalalignment='center', verticalalignment='bottom',
                    color='white', fontsize=22)
            ax.text(image_width-85, (image_height + 20)/2, 'R', 
                    horizontalalignment='center', verticalalignment='top', 
                    color='white', fontsize=22)

       
       
    def plot_moving_window(self, timestamp,  windowsize, ax, measurement, colour, title, xtitle):
        """
        timestamp = timestamp object corresponding to current frame
        windowsize = full span of the plot in seconds
        ax = subplot name
        measurement = thing to plot
        colour = line colour
        title = y axis title
        """
        timestamp = pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
        window_td = datetime.timedelta(0,windowsize,0)
        first_ts = (timestamp - window_td)
        last_ts = (timestamp + window_td) 
        window_data = self._data[(self._data.index >= first_ts) & (self._data.index <= last_ts)]
        #print (window_data['Timestamp'] - timestamp).values, '\t', window_data[measurement].values
        x_values = (window_data['Timestamp'] - timestamp).values / np.timedelta64(1,'s')
        y_values = window_data[measurement].values
        plt.fill_between(x_values, y_values,  color=colour)#, linewidth=2)
        plt.axvline((0), color='#00FF00', linewidth=2)
        #call before this function: AXNAME = fig.add_subplot(x,x,x) \n fig.add_axes(AXNAME)
        
        #ax.set_xlim(first_ts, last_ts)
        ax.set_ylim(0.85*(min(self._data[measurement])),1.15*(max(self._data[measurement])))
        ax.set_ylabel(title , fontsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8)
        if xtitle == 'w_titles': 
            ax.set_xlabel('Time (s)', fontsize=12)
        
    def plot_extending_line(self, timestamp, ax, measurement, colour, title, left_bound, right_bound, axtitle): 
        
        """
        left_bound & right_bound are integers corresponding to seconds since t0.
        """
        timestamp = pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
        
        axis_data = self._data[(self._data.synced_time >= np.timedelta64(left_bound, 's')) & 
                               (self._data.synced_time <= np.timedelta64(right_bound, 's'))]
        visible_data = axis_data[(axis_data.index <= timestamp)]
        
        #laser_values = self.******
        
        x_values = (visible_data['synced_time'] / np.timedelta64(1,'s')).values
        y_values = visible_data[measurement].values
        
        plt.plot(x_values, y_values,  color=colour, linewidth=3)
        ax.set_xlim(min(axis_data['synced_time'] / np.timedelta64(1, 's')) , max(axis_data['synced_time'] / np.timedelta64(1, 's')))  
         
        ax.set_ylim(0.85*(min(axis_data[measurement].dropna())),1.15*(max(axis_data[measurement].dropna())))
        ax.set_ylabel(title , fontsize=22)
        if axtitle == 'w_titles':
            ax.set_xlabel('Time (s)', fontsize=22)
            ax.tick_params(axis='y', which='major', labelsize=12)  


        plt.axvline((visible_data['synced_time'][-1] / np.timedelta64(1,'s')), color='#00FF00', linewidth=2)  
        """
        # REMOVE THE HACK BELOW HERE!
        
        STIM_LIST = [[0,1],[4,5],[8,9],[12,13],[16,17],[20,21],[24,25],[28,29],[32,33],[36,37]]
        for ons, offs in STIM_LIST:
            ax.axvspan(ons, offs,  facecolor='red', linewidth=0, edgecolor=None, alpha=0.3, zorder=10)
        
        """
        laser_1 = collections.BrokenBarHCollection.span_where((axis_data.synced_time.values/ np.timedelta64(1,'s')), 
                    ymin=0.85*(min(axis_data[measurement].dropna())),ymax=1.15*(max(axis_data[measurement].dropna())), 
                    where=axis_data['Laser1_state'] > 0.0, facecolor='k', edgecolor=None,
                    alpha=0.1, zorder=10) #green b2ffb2
        ax.add_collection(laser_1)
        

        laser_2 = collections.BrokenBarHCollection.span_where((axis_data.synced_time.values/ np.timedelta64(1,'s')), 
                    ymin=0.85*(min(axis_data[measurement].dropna())),ymax=1.15*(max(axis_data[measurement].dropna())), 
                    where=axis_data['Laser2_state'] > 0.0, facecolor='r', 
                    edgecolor='r', alpha=0.2, zorder=11) #green b2ffb2
        ax.add_collection(laser_2)
        
        
        
