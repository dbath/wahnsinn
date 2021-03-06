from cv2 import cv
import glob
import numpy as np
import pandas as pd
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import random
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm




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
    
    
    
height = 150
width = 150
_data = np.zeros((height, width), np.float32)

for x in glob.glob('/nearline/dickson/bathd/FlyMAD_data_archive/P1_adt2_sequential_activation/150513/*zoom*/*.fmf'):
    if os.path.exists(x.rsplit('/',1)[0]+'/tracking_info.pickle') ==True:
        print x
        _data = get_heads(x)
        

plt.imshow(_data, cmap = cm.Greys_r)
plt.axis('off')
plt.show()

mask = _data / _data.max()

np.save('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/template_files/head_template.npy', mask)

print 'saved template'
