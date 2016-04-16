
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

_right_bound = 600 #plots number of seconds after stim onset.



def make_panel(flyInstance, row, column, _frame_number, FLY_LABEL):

        frame,timestamp = flyInstance.get_frame((_frame_number + flyInstance._Tzero + -440))     #1800 starts @120sec
        
        jaaba_datum = flyInstance._data[flyInstance._data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
        
        
        wide_framenumber = flyInstance.get_frame_number_at_or_before_timestamp(flyInstance._widefmf, timestamp)
        wideframe, wideTS = flyInstance.get_wide(wide_framenumber)
        
        ax1 = plt.subplot2grid((60,200), (1,100), colspan=100, rowspan=30)
        
        fig.add_axes(ax1)
        flyInstance.plot_zoom(frame, timestamp, cm.Greys_r, jaaba_datum, ax1)
        
        #ax4 is widefield view, furthest left
        
        ax4 = plt.subplot2grid((60,200), (1,0), colspan=100, rowspan=30)
        fig.add_axes(ax4)
        flyInstance.plot_wide(wideframe, cm.Greys_r, ax4, 100) 
        
        #ax2 = fig.add_subplot(2,1,2)
        ax2 = plt.subplot2grid((60,200), (31, 6), colspan=194, rowspan=10)
        
        fig.add_axes(ax2)
        flyInstance.plot_extending_line(timestamp, ax2, 'maxWingAngle', '#A31919', 'Wing Angle (rad)', _left_bound, _right_bound, 'no_titles')
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        ax3 = plt.subplot2grid((60,200), (42,6), colspan=194, rowspan=10)
        
        fig.add_axes(ax3)
        flyInstance.plot_extending_line(timestamp, ax3, 'dtarget', '#0033CC', 'Distance (mm)', _left_bound, _right_bound, 'w_titles')        
        
        #ax5 is a vertical label on the left side
        
        ax5 = plt.subplot2grid((60,2), (0, 0), colspan=2, rowspan=1)
        
        t = ax5.text(0.5,0.5, FLY_LABEL, bbox={'facecolor':'w','edgecolor':'w', 'alpha':0.6, 'pad':10}, 
                     rotation=0, va='center', ha='center',
                     color='k', fontsize=42)
                     
        plt.setp(ax5.get_yticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        ax5.set_xticks([]); ax5.set_yticks([])
        ax5.axis('off')
        fig.add_axes(ax5)
        plt.tight_layout()
        #plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.25, hspace=0.15)


    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fmf', type=str, required=True,
                        help='path to fmf directory') 
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save directory') 
    parser.add_argument('--label', type=str, required=False, default="Control",
                        help='label for top data') 
    args = parser.parse_args()
    
    fmf = args.fmf
    savedir = args.savedir  
    
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

    fly = data_movies.FlyPanel(fmf, savedir, utilities.parse_fmftime(fmf)[0])
    
        
    image_height = (fly.get_frame(0)[0].shape[0] )*2
    
    image_width = fly.get_frame(0)[0].shape[1] *2:
    if not os.path.exists(savedir + '/temp_png):
        os.makedirs(savedir + '/temp_png) 
    for frame_number in range(fly.get_n_frames()):
           
        if os.path.exists(savedir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        fig = plt.figure(figsize=(image_width/100, image_height/100), dpi=200.399 )
           
        make_panel(fly, 0, 0, frame_number, args.label)
        #plt.show()
        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')  
        print 'added png: ', frame_number
    utilities.call_command("ffmpeg -f image2 -r 15 -i "+ savedir + "/temp_png/_tmp%05d.png -vf scale=iw/2:-1 -vcodec mpeg4 -b 8000k -y " + savedir[:-1] + ".mp4;")
    utilities.call_command("rm -r " + savedir)

    utilities.sendMail('bathd@janelia.hhmi.org','movie is finished', ('Your awesome new video has finished.'))
    
    
    
    """
    
    ffmpeg -f image2 -r 15 -i temp_png/_tmp%05d.png -vf 'scale=iw/2:-1' -vcodec mpeg4 -b 8000k -y 'movie.mp4'

    ffmpeg -f image2 -r 15 -i temp_png/_tmp%05d.png -vf 'scale=iw/2:-1' -vcodec mpeg4 -b 2000k -y 'movie.mp4'

"""
    
    
