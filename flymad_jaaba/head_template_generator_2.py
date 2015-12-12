from cv2 import cv
import glob
import numpy as np
import pandas as pd
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import random
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import flymad_jaaba.image_matcher as image_matcher




def subimage(image, centre, theta, width, height):
    theta *= -1.0
    output_image = cv.fromarray(np.zeros((height, width), np.float32))
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                        [np.sin(theta), np.cos(theta), centre[1]]])
    map_matrix_cv = cv.fromarray(mapping)
    cv.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
    return output_image
    


def get_heads(FMF_FILE):
    fmf = FMF.FlyMovie(FMF_FILE)
    for framenumber in random.sample(range(0,fmf.get_n_frames()), 2000):
        try:
            frame, timestamp = fmf.get_frame(framenumber)
            df = pd.read_pickle(FMF_FILE.rsplit('/',1)[0]+'/tracking_info.pickle' )
            hx = df.ix[framenumber].c_head_location_x
            hy = df.ix[framenumber].c_head_location_y
            centre = (int(hx), int(hy))
            theta = np.radians(df.ix[framenumber].d_bodyAxis)
            patch = subimage(cv.fromarray(frame), centre, theta, width, height)
            data = _data + np.array(patch)
        except:
            continue
    return data
    
def get_head_image(FMF_FILE, framenumber):
    fmf = FMF.FlyMovie(FMF_FILE)
    frame, timestamp = fmf.get_frame(framenumber)
    df = pd.read_pickle(FMF_FILE.rsplit('/',1)[0]+'/tracking_info.pickle' )
    hx = df.ix[framenumber].c_head_location_x
    hy = df.ix[framenumber].c_head_location_y
    centre = (int(hx), int(hy))
    theta = np.radians(df.ix[framenumber].d_bodyAxis)
    patch = subimage(cv.fromarray(frame), centre, theta, width, height)
    return np.array(patch)

def get_head_image_looper(_frame, _vals):
    hx = _vals.c_head_location_x
    hy = _vals.c_head_location_y
    centre = (int(hx), int(hy))
    theta = np.radians(_vals.d_bodyAxis)
    patch = subimage(cv.fromarray(_frame), centre, theta, width, height)
    return np.array(patch)

    
height = 150
width = 150

template = np.load('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/template_files/head_template_gen2.npy')
t2 = (template*255.0).astype(np.uint8)
accumulator = np.zeros((height, width), np.float32)

for x in glob.glob('/nearline/dickson/bathd/FlyMAD_data_archive/P1_adt2_sequential_activation/150513/*zoom*/*.fmf'):
    if os.path.exists(x.rsplit('/',1)[0]+'/tracking_info.pickle') ==True:
        
        print x
        
        fmf = FMF.FlyMovie(x)
        df = pd.read_pickle(x.rsplit('/',1)[0]+'/tracking_info.pickle')
         
        for framenumber in random.sample(range(0,fmf.get_n_frames()), 100): #range(fmf.get_n_frames()):
            frame, timestamp = fmf.get_frame(framenumber)
            
            try:
                _data = get_head_image_looper(frame, df.ix[framenumber])
            except:
                print "head grabbing failed:   ", framenumber
            try:
                kp_pairs = image_matcher.match_images(t2, _data.astype(np.uint8))
                
                if len(kp_pairs) > 0:
                    accumulator = accumulator + _data
                    print "*********MATCHED**********", framenumber
            except:
                print 'matching failed: ', framenumber

plt.imshow(accumulator, cmap = cm.Greys_r)
plt.axis('off')
plt.show()

mask = accumulator / accumulator.max()

np.save('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/template_files/head_template_gen3.npy', mask)

print 'saved template'
