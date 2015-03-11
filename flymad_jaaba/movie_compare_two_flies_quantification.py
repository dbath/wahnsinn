
import flymad_jaaba.utilities as utilities
import flymad_jaaba.data_movies as data_movies
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

#for plotting moving window:

_windowsize = 60 #for sixty seconds before and after present

#for plotting extending line:

_left_bound = -30  #plots 30 seconds before stim

_right_bound = 320 #plots number of seconds after stim onset.



def make_panel(flyInstance, row, column, _frame_number):

        #ax1 = fig.add_subplot(2,1,1)
        ax1 = plt.subplot2grid((60,6), (row,0), colspan=2, rowspan=30)
        
        fig.add_axes(ax1)
        frame,timestamp = flyInstance.get_frame((_frame_number + flyInstance._Tzero + 1125))     #1800 starts @120sec
        
        jaaba_datum = flyInstance._data[flyInstance._data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
        flyInstance.plot_zoom(frame, timestamp, cm.Greys_r, jaaba_datum, ax1)
        
        #ax2 = fig.add_subplot(2,1,2)
        ax2 = plt.subplot2grid((60,6), (row+3, 2), colspan=4, rowspan=10)
        
        fig.add_axes(ax2)
        flyInstance.plot_extending_line(timestamp, ax2, 'maxWingAngle', '#A31919', 'Wing Angle (rad)', _left_bound, _right_bound, 'no_titles')
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        ax3 = plt.subplot2grid((60,6), (row+14,2), colspan=4, rowspan=10)
        
        fig.add_axes(ax3)
        flyInstance.plot_extending_line(timestamp, ax3, 'dtarget', '#0033CC', 'Distance (px)', _left_bound, _right_bound, 'w_titles')        
        
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.25, hspace=0.15)


    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--topfmf', type=str, required=True,
                        help='path to fmf') 
    parser.add_argument('--bottomfmf', type=str, required=True,
                        help='path to fmf') 
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save directory') 
    args = parser.parse_args()
    
    top_fmf = args.topfmf
    bottom_fmf = args.bottomfmf
    savedir = args.savedir  
    
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

    ctrlfly = data_movies.FlyPanel(top_fmf, savedir, utilities.parse_fmftime(top_fmf)[0])
    
    expfly = data_movies.FlyPanel(bottom_fmf, savedir, utilities.parse_fmftime(bottom_fmf)[0])
        
    image_height = ctrlfly.get_frame(0)[0].shape[0] + expfly.get_frame(0)[0].shape[0]
    
    image_width = expfly.get_frame(0)[0].shape[1] *3
    for frame_number in range(1875):#ctrlfly.get_n_frames()):
           
        if os.path.exists(savedir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        fig = plt.figure(figsize=(image_width/100, image_height/100), dpi=200.399 )
           
        make_panel(ctrlfly, 0, 0, frame_number)
        make_panel(expfly, 30, 0, frame_number) 
        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')  
        print 'added png: ', frame_number
    for frame_number in range(3225,3500):#ctrlfly.get_n_frames()):
           
        if os.path.exists(savedir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        fig = plt.figure(figsize=(image_width/100, image_height/100), dpi=200.399 )
           
        make_panel(ctrlfly, 0, 0, frame_number)
        make_panel(expfly, 30, 0, frame_number) 
        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')  
        print 'added png: ', frame_number

    utilities.sendMail('bathd@janelia.hhmi.org','movie is finished', ('Your awesome new video has finished.'))
    
    
    
    """
    
    ffmpeg -f image2 -r 15 -i _tmp%05d.png -vf 'scale=iw/2:-1' -vcodec yuv420p -b 8000k -y '/media/DBATH_6/150226/movie.mp4'


"""
    
    
