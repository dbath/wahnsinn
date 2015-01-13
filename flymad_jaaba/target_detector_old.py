
import os.path
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



def generate_background_image(fmf_file, sample_rate, file_path):# wide_fmf in FMF format, sample rate: n for every nth frame
    wide_fmf = FMF.FlyMovie(fmf_file)
    frame0, _ = wide_fmf.get_frame(0)
    image_width, image_height = frame0.shape
    acc=np.zeros((image_width, image_height), np.float32) # 32 bit accumulator
    for frame_number in range(0, wide_fmf.get_n_frames(), sample_rate):
        frame, timestamp = wide_fmf.get_frame(frame_number)
        acc = np.maximum.reduce([frame, acc])
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image_height, image_width)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(acc, cmap = cm.Greys_r)
    plt.savefig(file_path, dpi=1)
    plt.close('all')
    return
    
def detect_targets(path_to_fmf, path_to_bg_images, path_to_target_images):
    fmf_id = path_to_fmf.split('/')[-1].split('.fmf')[0] + '.png'
    if not os.path.exists(path_to_bg_images + fmf_id):
        generate_background_image(path_to_fmf, 500, (path_to_bg_images + fmf_id))
    im = cv2.imread(path_to_bg_images + fmf_id)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    #BINARIZE AND DETECT CONTOURS FOR TARGETS
    ret, thresh = cv2.threshold(imgray, 128, 255, 0)
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    
    contours_targets, hierarchy = cv2.findContours(eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)    
    targets = []
    for c in contours_targets:
        area = cv2.contourArea(c)
        if (area >= 50) and (area <=300):
            M = cv2.moments(c)
            #cx,cy, radius = int(M['m10']/M['m00']), int(M['m01']/M['m00']), 5
            (cx,cy), radius = cv2.minEnclosingCircle(c)
            cv2.circle(im,(int(cx),int(cy)),int(radius),(255,0,0),2)
            targets.append([cx,cy, radius])
    
    """
    #BINARIZE AND DETECT CONTOURS FOR ARENA 
    ret, thresh_arena = cv2.threshold(imgray, 100, 255, 0)
    kernel = np.ones((5,5),np.uint8)
    eroded_arena = cv2.erode(thresh_arena, kernel, iterations=1)
    dilated_arena = cv2.dilate(eroded_arena, kernel, iterations=1)
    contours_arena, hierarchy = cv2.findContours(dilated_arena, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)    
    
    for c in contours_arena:
        area = cv2.contourArea(c)
        if area >=5000:
            M = cv2.moments(c)
            (cx,cy), radius = cv2.minEnclosingCircle(c)
            cv2.circle(im,(int(cx),int(cy)),int(radius),(0,255,0),2)
    """        
    cv2.imshow(fmf_id, im)
    #cv2.waitKey(0)
    cv2.imwrite((path_to_target_images + fmf_id), im)
    cv2.destroyAllWindows()
    return targets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--widefmf', type=str, required=True,
                        help='path to fmf') 
    parser.add_argument('--dump', type=str, required=True,
                        help='directory to store data')

    args = parser.parse_args()

    VIDEO_DIR = args.widefmf
    DUMP = args.dump

    if not os.path.exists(DUMP + '/background_images/' ) ==True:
        os.makedirs(DUMP + '/background_images/' )
    if not os.path.exists(DUMP + '/target_images/' ) ==True:
        os.makedirs(DUMP + '/target_images/' )



    for _fmf_file in glob.glob(VIDEO_DIR + '/' + '*wide*' + '*.fmf'):

        print 'processing: ', _fmf_file
        _path_to_bg_images = DUMP + '/background_images/' 
        _path_to_target_images = DUMP + '/target_images/' 
        detect_targets(_fmf_file, _path_to_bg_images, _path_to_target_images)
        
    
    
    
    
