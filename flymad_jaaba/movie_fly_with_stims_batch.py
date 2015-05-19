
import flymad_jaaba.utilities as utilities
import flymad_jaaba.bag_movies as bag_movies
import os
import glob
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

_right_bound = 200 #plots 200 seconds after stim onset.

FPS = 15

def make_panel(flyInstance, row, column, _frame_number, _fig, _ax):
        
        _fig.add_axes(_ax)
        frame,timestamp = flyInstance.get_frame((_frame_number))# TEMPORARY + flyInstance._Tzero - 150))     #1800 starts @-10sec
        
        jaaba_datum = flyInstance._data[flyInstance._data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
        
        #flyInstance.set_overlays(overlays_value)
        
        flyInstance.plot_zoom(frame, timestamp, cm.Greys_r, jaaba_datum, _ax)
        
            
def make_pngs(zoom_fmf, savedir, overlays_value):

    
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

    vid = bag_movies.FlyPanel(zoom_fmf, savedir, utilities.parse_fmftime(zoom_fmf)[0], overlays_value)

    
    image_height = vid.get_frame(0)[0].shape[0]
    
    image_width = vid.get_frame(0)[0].shape[1]
    
    for frame_number in np.arange(vid.get_n_frames()):
                
        if os.path.exists(savedir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        
        fig = plt.figure(figsize=(image_width/100, image_height*1.2/100), dpi=200.399, frameon=False )
        ax = plt.subplot2grid((10,10), (0,0), colspan=10, rowspan=10)
           
        make_panel(vid, 0, 0, frame_number, fig, ax)
        
        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputdir', type=str, required=True,
                        help='path parent dir ex: /foo/bar/150513') 
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save directory') 
    #parser.add_argument('--overlays', type=bool, required=False, default=False, help='turn on overlays to plot wing angles and distance bars')
    args = parser.parse_args()
    
    for x in glob.glob(args.inputdir + '/*zoom*'):
        png_stash = args.savedir + '/' + x.split('/')[-1].split('.')[0]
        if not os.path.exists(png_stash):
            os.makedirs(png_stash)  
        make_pngs(x, png_stash+'/', False)
        print 'generated PNG files for:', x
        
        
    utilities.sendMail('bathd@janelia.hhmi.org','movie is finished', ('Your awesome new video has finished.'))
