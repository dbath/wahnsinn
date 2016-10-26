
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
from multiprocessing import Process

#for plotting moving window:

_windowsize = 60 #for sixty seconds before and after present

#for plotting extending line:

_left_bound = -30  #plots 30 seconds before stim

_right_bound = 200 #plots 200 seconds after stim onset.

FPS = 15

def make_panel(flyInstance, row, column, _frame_number, fig):
        frame,timestamp = flyInstance.get_frame((_frame_number))# TEMPORARY + flyInstance._Tzero - 150))     #1800 starts @-10sec
        
        jaaba_datum = flyInstance._data[flyInstance._data['Timestamp'] == pd.to_datetime(timestamp, unit='s').tz_localize('UTC').tz_convert('US/Eastern')]
               
        
        ax = plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=1)
        fig.add_axes(ax)     
        flyInstance.plot_zoom(frame, timestamp, cm.Greys_r, jaaba_datum, ax)
        wide_framenumber = flyInstance.get_frame_number_at_or_before_timestamp(flyInstance._widefmf, timestamp)
        wideframe, wideTS = flyInstance.get_wide(wide_framenumber)
        ax2 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
        fig.add_axes(ax2)     
        flyInstance.plot_wide(wideframe, cm.Greys_r, ax2, 100) 
        fig.subplots_adjust(wspace=0.0)
            
def make_pngs(zoom_fmf, savedir, overlays_value):

    
    #os.system(ffmpeg -f image2 -r 15 -i _tmp%05d.png -vcodec mpeg4 -y (_VIDEO_DIR + '/flymad_annotated.mp4'))

    vid = bag_movies.FlyPanel(zoom_fmf, savedir, utilities.parse_fmftime(zoom_fmf)[0], overlays_value, args.lasermode)

    
    image_height = vid.get_frame(0)[0].shape[0]
    
    image_width = vid.get_frame(0)[0].shape[1]
    
    for frame_number in np.arange(vid.get_n_frames()):
                
        if os.path.exists(savedir + '/temp_png/_tmp%05d.png'%(frame_number)):
            continue
        fig = plt.figure(figsize=(image_width*2/100, image_height/100), dpi=200.399, frameon=False )
        make_panel(vid, 0, 0, frame_number, fig)
        
        plt.savefig(savedir + '/temp_png/_tmp%05d.png'%(frame_number), bbox_inches='tight', pad_inches=0)
        plt.close('all')    
    utilities.call_command("ffmpeg -f image2 -r 15 -i "+ savedir + "/temp_png/_tmp%05d.png -vf scale=iw/2:-1 -vcodec mpeg4 -b 8000k -y " + savedir[:-1] + ".mp4;")
    utilities.call_command("rm -r " + savedir)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inputdir', type=str, required=True,
                        help='path parent dir ex: /foo/bar/150513') 
    parser.add_argument('--savedir', type=str, required=True,
                        help='path to save directory') 
    parser.add_argument('--lasermode', type=str, required=False, default='state',
                        help='which laser value to use, "state" or "current" or "nolaser"')
    #parser.add_argument('--overlays', type=bool, required=False, default=False, help='turn on overlays to plot wing angles and distance bars')
    args = parser.parse_args()
    

    threadcount = 0
    _filelist = []
    
    for _directory in glob.glob(args.inputdir + '/*zoom*/*.fmf'):
        _filelist.append(_directory)
    for x in np.arange(len(_filelist)):    
        png_stash = args.savedir + '/' + _filelist[x].split('/')[-1].split('.')[0]
        if not os.path.exists(png_stash + '.mp4'):
            if not os.path.exists(png_stash):
                os.makedirs(png_stash)  
            p = Process(target=make_pngs, args=(_filelist[x], png_stash+'/',False))
            p.start()
            print _filelist[x]
            threadcount +=1
            
            if p.is_alive():
                if threadcount >=8:
                    threadcount = 0
                    p.join()
                elif _filelist[x] == _filelist[-1]:
                    threadcount=0
                    p.join()        
        
    utilities.sendMail('bathd@janelia.hhmi.org','movie is finished', ('Your awesome new video has finished.'))
