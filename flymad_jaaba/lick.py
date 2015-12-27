from cv2 import cv
import cv2
import glob
import numpy as np
import pandas as pd
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import random
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse




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
    df = pd.read_pickle(FMF_FILE.rsplit('/',1)[0]+'/tracking_info.pickle' )
    vals = []
    frames = []
    for framenumber in range(fmf.get_n_frames()):
        try:
            frame, timestamp = fmf.get_frame(framenumber)
            prob = get_proboscis(frame, df.ix[framenumber])
            vals.append(prob)
            frames.append(framenumber)
        except:
            continue
    valdf = pd.DataFrame({'Frame':frames,'Proboscis':vals})
    return valdf
    
def get_proboscis(frame, slyce):
    hx = slyce.c_head_location_x
    hy = slyce.c_head_location_y
    centre = (int(hx), int(hy))
    theta = np.radians(slyce.d_bodyAxis)
    patch = subimage(cv.fromarray(frame), centre, theta, width, height)
    data = np.array(patch)
    rat = data[75:95, 65:85].mean() / data[35:55, 65:85].mean()
    return rat    


def detect_proboscis(headImage):
    q = headImage / template
    im = cv2.cvtColor(q.astype(np.uint8), cv2.COLOR_GRAY2BGR) #must be uint8 array
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    ret2, mask = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY)
    dil= cv2.dilate(mask, kernel, iterations=1)
    eroded= cv2.erode(dil, kernel, iterations=1)
    contourImage = eroded.copy()
    probCont, hierarchy1 = cv2.findContours(contourImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidate_pros = pd.DataFrame({'area':[],'cx':[],'cy':[],'dCentre':[], 'tipAngle':[], 'tip_x':[], 'tip_y':[]})
    counter = 0
    for c in probCont:
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        if (area > 0) and (area < (width*height/2)):
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])  #CENTROID
            #distance_from_centre = np.sqrt((cx-width/2.0)**2 + (cy-height/2.0)**2)
            near, far = get_nearest_and_furthest_from_point(c, ((width/2),(height/2)))
            closest_d = np.sqrt((near[0]-width/2.0)**2 + (near[1]-height/2.0)**2)
            dMax = np.sqrt((far[0]-width/2.0)**2 + (far[1]-height/2.0)**2)
            (TIP_x, TIP_y) = get_max_y_value(c)
            tipAngle = abs(compute_axis_from_points((width/2,height/2), (TIP_x,TIP_y))[0])
            
            
            candidate_pros.loc[counter] = [area,cx, cy , closest_d, tipAngle, TIP_x, TIP_y]
            counter +=1
    try:
        candidate_pros['score'] = 0.0
        candidate_pros['length'] = np.sqrt((candidate_pros.tip_x-width/2.0)**2 + (candidate_pros.tip_y-height/2.0)**2)
        #candidate_pros.loc[candidate_pros.tip_y <=(height/2), 'score'] -= 100
        candidate_pros.loc[candidate_pros.tipAngle.argmax(), 'score'] +=1
        candidate_pros.loc[candidate_pros.area.argmax(), 'score'] +=1
        candidate_pros.loc[candidate_pros.dCentre.argmin(), 'score'] +=1
        candidate_pros.loc[abs(candidate_pros.cx - width/2).argmin(), 'score'] +=1
        proboscis = candidate_pros[(candidate_pros.tip_y >=(height/2))
                                  & (candidate_pros.tipAngle >= 2.0)
                                  & (candidate_pros.length <= 50.0)
                                  & (candidate_pros.dCentre <= 10.0)
                                  ].ix[candidate_pros['score'].argmax()]
    except:
        proboscis = pd.Series({'area':0.0,'cx':width/2,'cy':height/2,'dCentre':0, 'length':0, 'tipAngle':0, 'tip_x':width/2, 'tip_y':height/2})
    return proboscis
    
def get_nearest_and_furthest_from_point(hullset, centroid):
    #PASS A SET OF POINTS DEFINING A SINGLE CONTOUR, IDEALLY OUTPUT FROM cv2.convexHull
    lowest_distance = 1000000
    lowest_coords = (0,0)
    highest_distance = 0
    highest_coords = (0,0)
    for a in hullset:
        b = (a[0][0], a[0][1])
        distance = np.sqrt((b[0]-width/2.0)**2 + (b[1]-height/2.0)**2)
        if distance > highest_distance:
            highest_coords = b
            highest_distance = distance
        if distance < lowest_distance:
            lowest_coords = b
            lowest_distance = distance 
    return lowest_coords, highest_coords

def compute_axis_from_points( POINT1, POINT2):
    if float(float(POINT1[0]) - float(POINT2[0]) ) == 0.0:
        XINT = POINT1[0]
        YINT = np.nan
        SLOPE = np.inf
    else:
        SLOPE = ( float(POINT1[1]) - float(POINT2[1])) / ( float(float(POINT1[0]) - float(POINT2[0]) ))
        YINT = POINT1[1] - (SLOPE*POINT1[0])
        if abs(SLOPE) >= 1000000:
            XINT = POINT1[0]
        elif SLOPE == 0.0:
            XINT = np.nan
        else:
            XINT = -1*YINT / SLOPE
    return SLOPE, YINT, XINT

def get_max_y_value(hullset):
    #PASS A SET OF POINTS DEFINING A SINGLE CONTOUR, IDEALLY OUTPUT FROM cv2.convexHull
    yvals = []
    xvals=[]
    coords = []
    for a in hullset:
        coords.append((a[0][0], a[0][1]))
        yvals.append(a[0][1])
    maxValue = [i for i, j in enumerate(yvals) if j == max(yvals)][0]
    return coords[maxValue]

        
def get_head_image(frame, slyce):
    try:
        hx = slyce.c_head_location_x
        hy = slyce.c_head_location_y
        centre = (int(hx), int(hy))
        theta = np.radians(slyce.d_bodyAxis)
        patch = subimage(cv.fromarray(frame), centre, theta, width, height)
    except:
        patch = subimage(cv.fromarray(frame), (502,502), 0, width, height)
    data = np.array(patch)
    rat = data[75:95, 65:85].mean() / data[35:55, 65:85].mean()
    return data
    


#-----------------------------
def track_proboscis(fmf_file, tracking_file):
    s = pd.DataFrame({'area':[],'cx':[],'cy':[],'dCentre':[],'length':[],'tipAngle':[],  'tip_x':[], 'tip_y':[]})
    fmf = FMF.FlyMovie(fmf_file)
    tracking = pd.read_pickle(tracking_file)
    
    if not os.path.exists(fmf_file.rsplit('/',1)[0] + '/proboscis_movie'):
        os.makedirs(fmf_file.rsplit('/',1)[0] + '/proboscis_movie')
    
    
    for framenumber in range(0,fmf.get_n_frames()):
        try:
            head = get_head_image(fmf.get_frame(framenumber)[0], tracking.ix[framenumber])
        except:
            head = get_head_image(fmf.get_frame(framenumber)[0], tracking.ix[0])
        pro = detect_proboscis(head)
        s.loc[framenumber] = pro
        
        imcopy = head.copy()
        try:
            cv2.line(imcopy, (75,75), (int(pro.tip_x),int(pro.tip_y)), (255,255,0),2)
        except:
            cv2.putText(imcopy, 'X' , (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.putText(imcopy, str(framenumber), (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imwrite(fmf_file.rsplit('/',1)[0] + '/proboscis_movie/_tmp%05d.png'%(framenumber), imcopy)
    s.to_pickle(fmf_file.rsplit('/',1)[0] + '/proboscis_data.pickle')
    return s
#-----------------------------
"""
s = pd.DataFrame({'area':[],'cx':[],'cy':[],'dCentre':[],'length':[],'tipAngle':[],  'tip_x':[], 'tip_y':[]})
for framenumber in range(0,fmf.get_n_frames()):
    try:
        head = get_head_image(fmf.get_frame(framenumber)[0], tracking.ix[framenumber])
    except:
        head = get_head_image(fmf.get_frame(framenumber)[0], tracking.ix[0])
    pro = detect_proboscis(head)
    s.loc[framenumber] = pro
    
    imcopy = head.copy()
    try:
        cv2.line(imcopy, (75,75), (int(pro.tip_x),int(pro.tip_y)), (255,255,0),2)
    except:
        cv2.putText(imcopy, 'X' , (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.putText(imcopy, str(framenumber), (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.imwrite('/groups/dickson/home/bathd/Desktop/proboscis_movie/_tmp%05d.png'%(framenumber), imcopy)
"""


height = 150
width = 150
_data = np.zeros((height, width), np.float32)
template = np.load('/groups/dickson/home/bathd/wahnsinn/flymad_jaaba/template_files/head_template_gen2.npy')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    args = parser.parse_args()

    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/' 
    for x in glob.glob(DATADIR + '*zoom*/*.fmf'):
        if not os.path.exists(x.rsplit('/',1)[0] + '/proboscis_data.pickle'):
            print "processing: ", x.split('/')[-1]
            try:
                _tracking = x.rsplit('/',1)[0] + '/tracking_info.pickle'
                _ = track_proboscis(x, _tracking)
            except:
                print "unable to process: ", x.split('/')[-1]
        
    
