import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2, cv
import roslib; roslib.load_manifest('flymad')
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import rosbag




def plot_means(framenumber):
    frame, timestamp = fmf.get_frame(framenumber)
    plt.imshow(frame)
    data = bagdf[(bagdf.Time <= timestamp +0.25) & (bagdf.Time >= timestamp -0.25)].mean()
    print data
    plt.plot(data.Hx, data.Hy, linewidth=0, marker='o', markersize=7, color='w', markeredgecolor='k', markeredgewidth=2)
    plt.plot(data.Bx, data.By, linewidth=0, marker='o', markersize=7, color='r', markeredgecolor='k', markeredgewidth=2)
    plt.show()



def plot_latest_values(framenumber):
    frame, timestamp = fmf.get_frame(framenumber)
    plt.imshow(frame)
    data = bagdf[(bagdf.Time <= timestamp)].index[-1]
    print data
    plt.plot(data.Hx, data.Hy, linewidth=0, marker='o', markersize=7, color='w', markeredgecolor='k', markeredgewidth=2)
    plt.plot(data.Bx, data.By, linewidth=0, marker='o', markersize=7, color='r', markeredgecolor='k', markeredgewidth=2)
    plt.show()


def plot_heading_line(framenumber):
    frame, timestamp = fmf.get_frame(framenumber)
    frame = devignette(frame)
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    s = ax.imshow(frame)
    #data = bagdf[(bagdf.Time >= timestamp)].index[0]
    data = bagdf[(bagdf.Time <= timestamp +0.03) & (bagdf.Time >= timestamp -0.03)].mean()
    print data
    if data.Hx > data.Bx:
        _xmin = data.Hx
        _xmax = len(frame[0])
    else:
        _xmin = 0
        _xmax = data.Hx
    if data.Hy > data.By:
        _ymin = data.Hy
        _ymax = len(frame)
    else:
        _ymin = 0
        _ymax = data.Hy
    
    if abs(data.slope) == np.inf:
        ax.axvline(x=data.Hx, ymin=_ymin, ymax=_ymax, color='k', linewidth=4)
        ax.axhline(y=data.Hy, xmin=0, xmax=len(frame[0]), color='k', linewidth=4, linestyle='--')
    else:
        x = np.linspace(_xmin,_xmax,2)
        y = (data.slope)*(x) + data.yint
        ax.plot(x,y,color='k', linewidth=4)
        x2 = np.linspace(0,len(frame[0]),2)
        y2 = (data.perp)*(x2) + data.perpInt
        ax.plot(x2,y2,color='k', linewidth=4, linestyle='--')


    cbaxes = fig.add_axes([0.9,0.2,0.025,0.60])
    cbar = plt.colorbar(s, cax = cbaxes )
    
    ax.set_ylim(0, len(frame))
    ax.set_xlim(0,len(frame[0]))
    ax.plot(data.Hx, data.Hy, linewidth=0, marker='o', markersize=7, color='w', markeredgecolor='k', markeredgewidth=2)
    ax.plot(data.Bx, data.By, linewidth=0, marker='o', markersize=7, color='r', markeredgecolor='k', markeredgewidth=2)
    plt.show()


def devignette(frame):
    V_coeff =[ 0.608421,0.000660594,0.00071838,
               -6.83654e-07,2.29008e-07,-6.11814e-07,
               -8.79999e-11,-1.63231e-10,-2.10072e-11,-2.10298e-10]

    mask = np.ones([len(frame[0]), len(frame)])

    xx, yy = np.meshgrid(np.arange(0,len(frame[0]),1), np.arange(0,len(frame),1))

    V_fit = mask*V_coeff[0] + xx*V_coeff[1] + yy*V_coeff[2] + xx**2*V_coeff[3] + xx*yy*V_coeff[4] + yy**2*V_coeff[5] + xx**3*V_coeff[6] + xx**2*yy*V_coeff[7] + xx*yy**2*V_coeff[8] + yy**3*V_coeff[9]                                         

    devign = (frame / V_fit).astype(np.uint8)
    
    return devign



def get_data_from_bag(bagfile):
    bag = rosbag.Bag(bagfile)
    head_x = []
    head_y = []
    body_x = []
    body_y = []
    times = []
    for topic, msg, t in bag.read_messages('/flymad/laser_head_delta'):
        head_x.append(msg.head_x)
        head_y.append(msg.head_y)
        body_x.append(msg.body_x)
        body_y.append(msg.body_y)
        times.append((t.secs + t.nsecs*1e-9))
        
    newdf = pd.DataFrame({'Time':times, 
                          'Hx':np.around(head_x), 
                          'Hy':np.around(head_y),
                          'Bx':np.around(body_x), 
                          'By':np.around(body_y)})
                          
    newdf = newdf[newdf.Hx < 1000000]    #failed detection msgs are filled with value 1e6.
    return newdf

def compute_body_axes(newdf):
    # calculate 'norm' the distance between body and head points:
    newdf['norm'] = np.sqrt((newdf.Hx-newdf.Bx)**2 + (newdf.Hy-newdf.By)**2)

    newdf['slope'] = (newdf.Hy-newdf.By) / (newdf.Hx-newdf.Bx)
    newdf['perp'] = -1*(newdf.Hx-newdf.Bx) / (newdf.Hy-newdf.By)
    newdf['yint'] = newdf.Hy - (newdf.slope * newdf.Hx)
    newdf['perpInt'] = newdf.Hy - (newdf.perp * newdf.Hx)  
    return newdf

fmf_file = '/media/DBATH_7/150707/R3lexTNT26992L-SS01538-redstim_zoom_20150713_101516/R3lexTNT26992L-SS01538-redstim_zoom_20150713_101516.fmf'
bag_fn = '/media/DBATH_7/150707/BAGS/rosbagOut_2015-07-13-10-15-14.bag'

bagdf = get_data_from_bag(bag_fn)
bagdf = compute_body_axes(bagdf)
fmf = FMF.FlyMovie(fmf_file)


frame, timestamp = fmf.get_frame(4567)

frame = devignette(frame)

im = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #must be uint8 array
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret1, thresh1 = cv2.threshold(imgray, 60,255,cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(imgray, 110,1,cv2.THRESH_BINARY_INV)
test = thresh1*thresh2
kernel = np.ones((5,5),np.uint8)
eroded = cv2.erode(test, kernel, iterations=2)
dilated = cv2.dilate(eroded, kernel, iterations=1)

contours_targets, hierarchy = cv2.findContours(eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)    
targets = []
hulls = []
counter=0
for c in contours_targets:
    area = cv2.contourArea(c)
    #TARGETS MUST BE APPROPRIATE SIZE
    if (area >= 5000):
        counter+=1
        (x, y, w, h) = cv2.boundingRect(c)
        targets.append([x,y,w,h])
        hulls.append(cv2.convexHull(c))
        
for x in targets:
    plt.Rectangle((x[0],x[1]),x[2],x[3], color='k')

def get_distance_between_coords(A, B):
    return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    
   
def get_centroid_and_head(_timestamp):
    data = bagdf[bagdf.Time >= _timestamp].iloc[0]
    return (data.Bx, data.By), (data.Hx, data.Hy)

def get_nearest_and_furthest_from_centroid(hullset, centroid):
    #PASS A SET OF POINTS DEFINING A SINGLE CONTOUR, IDEALLY OUTPUT FROM cv2.convexHull
    lowest_distance = 1000000
    lowest_coords = (0,0)
    highest_distance = 0
    highest_coords = (0,0)
    for a in hullset:
        print a[0]
        b = (a[0][0], a[0][1])
        distance = get_distance_between_coords(centroid, b)
        if distance > highest_distance:
            highest_coords = b
            highest_distance = distance
        if distance < lowest_distance:
            lowest_coords = b
            lowest_distance = distance 
    return lowest_coords, highest_coords
    
def plot_key_points(timestamp, hullsetnum, _ax):
    body, head = get_centroid_and_head(timestamp)
    near, far = get_nearest_and_furthest_from_centroid(hulls[hullsetnum], body)

    _ax.scatter(head[0], head[1], color='w', marker='o')
    _ax.scatter(body[0], body[1], color='r', marker='o')
    _ax.scatter(near[0], near[1], color='k', marker='o')
    _ax.scatter(far[0], far[1], color='b', marker='o')

def buildit():
    global fig
    fig = plt.figure()
    global ax 
    ax = fig.add_subplot(1,1,1)
    ax.imshow(imgray)
    

