import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import cv2, cv
import roslib; roslib.load_manifest('flymad')
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import rosbag
import flymad_jaaba.utilities as utilities
import flymad_jaaba.target_detector as target_detector
import math

class WingDetector(object):

    def __init__(self, zoomFMF_filepath, bag_filepath, dTarget, arena_centre, tempdir=None ):
        
        
        self.fmf_file = zoomFMF_filepath
        self.fmf = FMF.FlyMovie(self.fmf_file)
        
        self.bag_fn = bag_filepath
        self.bagdf = self.get_data_from_bag(self.bag_fn)
        self.bagdf = self.compute_body_axes(self.bagdf)
        self.positions = self.get_positions_from_bag(self.bag_fn)
        self.dTarget = dTarget
        (self.arena_centre) = arena_centre
        if tempdir is not None:
            self.saveImage = True
            if tempdir[-1] == '/':
                pass
            else:
                tempdir = tempdir + '/'
            self._tempdir = tempdir
        else: self.saveImage = False
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
       
        self.wingData = DataFrame({'Timestamp':[],'Length':[],'Width':[], 'Left':[], 'Right':[]}, dtype=np.float64)            

    
    def execute(self):
        for frame_number in range(11000,self.fmf.get_n_frames()):
            self.detectWings(frame_number, self.saveImage)
        return
        
    def get_wingAngle(self, frame_number):
        t, L, R = self.detectWings(frame_number)
        return t, L, R
    
    def devignette(self, frame):
        V_coeff =[ 0.608421,0.000660594,0.00071838,
                   -6.83654e-07,2.29008e-07,-6.11814e-07,
                   -8.79999e-11,-1.63231e-10,-2.10072e-11,-2.10298e-10]

        mask = np.ones([len(frame[0]), len(frame)])

        xx, yy = np.meshgrid(np.arange(0,len(frame[0]),1), np.arange(0,len(frame),1))

        V_fit = mask*V_coeff[0] + xx*V_coeff[1] + yy*V_coeff[2] + xx**2*V_coeff[3] + xx*yy*V_coeff[4] + yy**2*V_coeff[5] + xx**3*V_coeff[6] + xx**2*yy*V_coeff[7] + xx*yy**2*V_coeff[8] + yy**3*V_coeff[9]                                         

        devign = (frame / V_fit).astype(np.uint8)
        
        return devign



    def get_data_from_bag(self, bagfile):
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
            
        newdf = pd.DataFrame({'Timestamp':times, 
                              'Hx':np.around(head_x), 
                              'Hy':np.around(head_y),
                              'Bx':np.around(body_x), 
                              'By':np.around(body_y)})
                              
        newdf = newdf[newdf.Hx < 1000000]    #failed detection msgs are filled with value 1e6.
        newdf = utilities.convert_timestamps(newdf)
        return newdf

    def get_positions_from_bag(self, bagfile):
        bag = rosbag.Bag(bagfile)
        px = []
        py = []
        times = []
        for topic, msg, t in bag.read_messages('/flymad/raw_2d_positions'):
            px.append(msg.points[0].x)
            py.append(msg.points[0].y)
            times.append((t.secs + t.nsecs*1e-9)) 
        newdf = pd.DataFrame({'Timestamp':times, 
                          'Px':np.around(px), 
                          'Py':np.around(py)})   
        newdf = utilities.convert_timestamps(newdf)
        return newdf

    def compute_body_axes(self, newdf):
        # calculate 'norm' the distance between body and head points:
        newdf['norm'] = np.sqrt((newdf.Hx-newdf.Bx)**2 + (newdf.Hy-newdf.By)**2)

        newdf['slope'] = (newdf.Hy-newdf.By) / (newdf.Hx-newdf.Bx)
        newdf['perp'] = -1*(newdf.Hx-newdf.Bx) / (newdf.Hy-newdf.By)
        newdf['yint'] = newdf.Hy - (newdf.slope * newdf.Hx)
        newdf['perpInt'] = newdf.Hy - (newdf.perp * newdf.Hx)  
        return newdf




    def detectWings(self, saveImage, debugging=False, framenumber=0):#, bodyThresh, wingThresh):

        frame, timestamp = self.fmf.get_frame(framenumber)
        
        timestamp_FMT = pd.to_datetime(timestamp, unit='s', utc=True).tz_convert('US/Eastern')
        
        #FLY FEATURES DERIVED FROM BAG FILE:
        centroid, head = self.get_centroid_and_head(timestamp_FMT)
        backPoint = tuple(sum(y) / len(y) for y in zip(centroid, head))
        headLine = self.compute_perpendicular_from_points(head, centroid)
        axisLine = self.compute_axis_from_points(head, centroid)
        
        #data = self.positions.loc[self.positions.index.asof(timestamp_FMT)]#[self.positions.Time >= timestamp].iloc[0]
        Px = self.positions.Px.asof(timestamp_FMT)    #SILLY HACK FOR 'MISMATCHING' INDICES. STUPID PANDAS.
        Py = self.positions.Py.asof(timestamp_FMT)
        distance = self.get_distance_between_coords((Px,Py), self.arena_centre)
        targ_dist = self.dTarget.asof(timestamp_FMT)
        
        bodyThresh, wingThresh, ellThresh = self.get_tracking_thresholds(timestamp_FMT, distance, targ_dist)
        
        # COMPUTER VISION:
        frame = self.devignette(frame)
        im = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #must be uint8 array
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((5,5),np.uint8)

        #FIT ELLIPSE TO BODY:
        ret2, body = cv2.threshold(imgray, ellThresh[0], 255, cv2.THRESH_BINARY)
        ellipseFitter = cv2.dilate(body, kernel, iterations=ellThresh[1])
        ellipseFitter = cv2.erode(body, kernel, iterations=ellThresh[2])
        contourImage = ellipseFitter.copy()
        bodyCont, hierarchy1 = cv2.findContours(contourImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        

        bodyEllipse=None
        
        for cnt in bodyCont:
            if cv2.contourArea(cnt) <=900000:
                if cv2.contourArea(cnt) >= 7000:
                    ellipse= cv2.fitEllipse(cnt)
                    if self.pointInEllipse(centroid[0],centroid[1],ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2]):
                        bodyEllipse = ellipse
                        slope = self.convert_ellipseAngle_to_slope(bodyEllipse[2])
                        yint = -1.0*slope*bodyEllipse[0][0] + bodyEllipse[0][1]
                        xint = (-1.0*yint / slope)
                        axisLine = slope, yint, xint
                        head = self.pointOfIntersection(headLine[0],headLine[1], axisLine[0], axisLine[1])
        if bodyEllipse == None:
            for cnt in bodyCont:
                if cv2.contourArea(cnt) <=900000:
                    if cv2.contourArea(cnt) >= 7000:
                        ellipse= cv2.fitEllipse(cnt)
                        bodyEllipse = ellipse
        
        if bodyEllipse == None:
            print "ERROR: cannot detect body ellipse in frame: ", framenumber
            
            imcopy = im.copy()
            cv2.putText(imcopy, "ERROR", (480,530), self.font, 1, (255,255,255), 3)
            if self.saveImage == True:
                cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy)  
            return timestamp, np.nan, np.nan, np.nan, np.nan                
        
        (f1, f2) = self.fociOfEllipse(bodyEllipse[0][0],bodyEllipse[0][1],bodyEllipse[1][0],bodyEllipse[1][1],bodyEllipse[2])

        if debugging == True:
            imcopy = im.copy()
            cv2.ellipse(imcopy,bodyEllipse,(255,255,255),3)
            cv2.circle(imcopy,(int(head[0]),int(head[1])),3,(0,255,0),-1)
            cv2.circle(imcopy,(int(centroid[0]),int(centroid[1])),3,(0,255,255),-1)
            cv2.circle(imcopy,(int(f1[0]),int(f1[1])),3,(255,255,0),-1)
            cv2.circle(imcopy,(int(f2[0]),int(f2[1])),3,(255,255,0),-1)
            head = self.get_nearest(head, [f1,f2])
            cv2.circle(imcopy,(int(head[0]),int(head[1])),5,(255,0,255),1)
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_00_ellipseFit.png', imcopy)
            cv2.destroyAllWindows
            
        
        head = self.get_nearest(head, [f1,f2])
        tail = self.get_furthest(head, [f1,f2])
        centroid = (bodyEllipse[0][0],bodyEllipse[0][1])
        backPoint = tuple(sum(y) / len(y) for y in zip(centroid, head))
        backPoint = tuple(sum(y) / len(y) for y in zip(centroid, backPoint))
        slope = self.convert_ellipseAngle_to_slope(bodyEllipse[2])
        yint = -1.0*slope*bodyEllipse[0][0] + bodyEllipse[0][1]
        xint = (-1.0*yint / slope)
        axisLine = slope, yint, xint

        centroid = bodyEllipse[0]
        headLine = self.compute_perpendicular_from_points(head, centroid)
        tailLine = self.compute_perpendicular_from_points(tail, centroid)
    

        #DEFINE bodyNotWings AS BODY PORTION PLUS LEGS ETC, USEFUL FOR FINDING WINGS.
        ret1, bodyNotWings = cv2.threshold(imgray, bodyThresh[0],255,cv2.THRESH_BINARY) 

        #DEFINE wings AS WINGS AND TARGETS BUT NOT BODY.   
        ret2, wings = cv2.threshold(imgray, wingThresh[0],1,cv2.THRESH_BINARY_INV)
        test = wings*bodyNotWings
        eroded = cv2.erode(test, kernel, iterations=wingThresh[1])
        dilated = cv2.dilate(eroded, kernel, iterations=wingThresh[2])
        dilatedCopy = dilated.copy()

        imcopy = im.copy()
        
        wingCont, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)    
        hulls = []
        
        if targ_dist <=20.0:
            headsideArea = 0.0
            headside = []
            tailsideArea = 0.0
            tailside = []
            for c in wingCont:
                area = cv2.contourArea(c)
                if area >= 1000:
                    M = cv2.moments(c)
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    if not self.check_laterality(centroid, (cx,cy), headLine[0], headLine[1], headLine[2]):
                        headsideArea = headsideArea + area
                        for x in c: headside.append(x.tolist())
                        #if debugging == True: cv2.drawContours(imcopy,c,3,(0,0,255),-1)
                    if not self.check_laterality(centroid, (cx,cy), tailLine[0], tailLine[1], tailLine[2]):
                        headsideArea = headsideArea + area
                        for x in c: tailside.append(x.tolist())
                        #if debugging == True: cv2.drawContours(imcopy,c,3,(0,125,255),-1)

            print len(tailside), len(headside)
            tailside = np.array(tailside).reshape((-1,1,2)).astype(np.int32)
            headside = np.array(headside).reshape((-1,1,2)).astype(np.int32)
            
            if len(tailside)*len(headside)>=1:
                if cv2.contourArea(tailside) > cv2.contourArea(headside):            
                #if tailsideArea > headsideArea:
                    head, tail = tail, head
                    headLine = tailLine
                    backPoint = tuple(sum(y) / len(y) for y in zip(centroid, head))
                    backPoint = tuple(sum(y) / len(y) for y in zip(centroid, backPoint))
                    print framenumber, ":   AXIS FLIPPED."
                    if debugging:
                        cv2.putText(imcopy, "FLIPPED", (120,120), self.font, 1, (255,255,255), 3)
                    
        for c in wingCont:
            area = cv2.contourArea(c)
            #WINGS MUST BE APPROPRIATE SIZE
            if (area >= 3000):
                M = cv2.moments(c)
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                #WINGS MUST BE BEHIND HEAD
                if self.check_laterality(centroid, (cx,cy), headLine[0], headLine[1], headLine[2]):
                    checkSpot = (c[0][0][0], c[0][0][1])
                    pointSet1 = []
                    pointSet2 = []
                    for x in c:
                        if self.check_laterality((x[0][0], x[0][1]), centroid, headLine[0], headLine[1], headLine[2]):
                            if self.check_laterality((x[0][0], x[0][1]), checkSpot, axisLine[0], axisLine[1], axisLine[2]):
                                pointSet1.append(x.tolist())
                            else:
                                pointSet2.append(x.tolist())
                    pointSet1 = np.array(pointSet1).reshape((-1,1,2)).astype(np.int32)
                    pointSet2 = np.array(pointSet2).reshape((-1,1,2)).astype(np.int32)
                    if (len(pointSet1) > 0):
                        if cv2.contourArea(pointSet1) >=(6000/(wingThresh[1]+1)):
                            if debugging == True: cv2.drawContours(imcopy,[pointSet1],0,(0,255,255),1)
                            hulls.append(cv2.convexHull(pointSet1))
                    if (len(pointSet2) > 0):
                        if cv2.contourArea(pointSet2) >=(6000/(wingThresh[1]+1)):
                            if debugging == True: cv2.drawContours(imcopy,[pointSet2],0,(255,0,0),1)
                            hulls.append(cv2.convexHull(pointSet2))
        
        wingTips = []
        for h in hulls: 

            near, far = self.get_nearest_and_furthest_from_centroid(h, centroid) 
            body_length = self.get_distance_between_coords(head,tail)
            abd_length =  self.get_distance_between_coords(backPoint, tail)   
            if self.get_distance_between_coords(near, centroid) <= 150:
                cv2.fillPoly(imcopy,h, (0,255,0))
                cv2.circle(imcopy, near, 3, (0,0,0), -1)
                cv2.circle(imcopy, far, 5, (255,255,255), -1)
                winglength = self.get_distance_between_coords(far, backPoint)
                if (winglength <= 2.0*(body_length)) and (winglength >= abd_length):
                    wingTips.append(far)
                    
        
        leftWingAngle = 0.0
        leftWing = tail
        rightWingAngle = 0.0
        rightWing = tail
        for wing in wingTips:
            THETA =  self.compute_angle_given_three_points(backPoint, wing, centroid)
            if THETA >= np.pi:
                THETA = THETA - 2.0*(np.pi)
            elif THETA <= -1.0*(np.pi):
                THETA = THETA + 2.0*(np.pi)
            if THETA >= 0.0:
                if THETA >= leftWingAngle:
                    leftWingAngle = THETA
                    leftWing = wing
            elif THETA <= 0.0:
                if THETA <= rightWingAngle:
                    rightWingAngle = THETA
                    rightWing = wing
        
        if debugging == True:            
            cv2.line(imcopy, (int(head[0]),int(head[1])), (int(tail[0]),int(tail[1])), (255,255,255), 1)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(leftWing[0]),int(leftWing[1])), (20,20,255),2)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(rightWing[0]),int(rightWing[1])), (20,255,20),2)
            cv2.circle(imcopy, (int(head[0]),int(head[1])), 3, (255,255,255), -1)
            cv2.circle(imcopy, (int(backPoint[0]),int(backPoint[1])), 5, (255,255,255), -1)
            #cv2.circle(imcopy, (int(centroid[0]),int(centroid[1])), 3, (255,0,255), -1)
            cv2.putText(imcopy, str(np.degrees(leftWingAngle)), (10,25), self.font, 1, (255,255,255), 3)
            cv2.putText(imcopy, str(np.degrees(rightWingAngle)), (512, 25), self.font, 1, (255,255,255), 3) 
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_05_results.png', imcopy)
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_01_bodyNotWings_'+str(bodyThresh[0])+'.png', bodyNotWings)
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_02_wings_'+str(wingThresh[0])+'.png', test)
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_03_eroded_'+str(wingThresh[1])+'.png', eroded)
            cv2.imwrite('/groups/dickson/home/bathd/Desktop/DEBUGGING/'+str(framenumber)+'_04_dilated_'+str(wingThresh[2])+'.png', dilatedCopy)
            
            
        if saveImage == True:
            imcopy = im.copy()
            cv2.line(imcopy, (int(head[0]),int(head[1])), (int(tail[0]),int(tail[1])), (255,255,255), 1)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(leftWing[0]),int(leftWing[1])), (20,20,255),2)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(rightWing[0]),int(rightWing[1])), (20,255,20),2)
            cv2.circle(imcopy, (int(head[0]),int(head[1])), 3, (255,255,255), -1)
            cv2.circle(imcopy, (int(backPoint[0]),int(backPoint[1])), 5, (255,255,255), -1)
            #cv2.circle(imcopy, (int(centroid[0]),int(centroid[1])), 3, (255,0,255), -1)
            cv2.putText(imcopy, str(np.degrees(leftWingAngle)), (10,25), self.font, 1, (255,255,255), 3)
            cv2.putText(imcopy, str(np.degrees(rightWingAngle)), (512, 25), self.font, 1, (255,255,255), 3) 

            cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy) 
        cv2.destroyAllWindows()
        
        LENGTH = bodyEllipse[1][1]# self.get_distance_between_coords(head, tail)
        WIDTH = bodyEllipse[1][0]
        
        print framenumber,  "\tL: ", np.degrees(leftWingAngle), '\tR: ', -1.0*(np.degrees(rightWingAngle)), '\t',distance, '\t', str(self.dTarget.asof(timestamp_FMT))
        
        self.wingData.loc[framenumber] = [timestamp, LENGTH, WIDTH, leftWingAngle, -1.0*rightWingAngle]
        
        return timestamp, LENGTH, WIDTH, leftWingAngle, -1.0*(rightWingAngle)

        
                
    def closestpair(self, L):
	    def square(x): return x*x
	    def sqdist(p,q): return square(p[0]-q[0])+square(p[1]-q[1])
	
	    # Work around ridiculous Python inability to change variables in outer scopes
	    # by storing a list "best", where best[0] = smallest sqdist found so far and
	    # best[1] = pair of points giving that value of sqdist.  Then best itself is never
	    # changed, but its elements best[0] and best[1] can be.
	    #
	    # We use the pair L[0],L[1] as our initial guess at a small distance.
	    best = [sqdist(L[0],L[1]), (L[0],L[1])]
	
	    # check whether pair (p,q) forms a closer pair than one seen already
	    def testpair(p,q):
		    d = sqdist(p,q)
		    if d < best[0]:
			    best[0] = d
			    best[1] = p,q
			
	    # merge two sorted lists by y-coordinate
	    def merge(A,B):
		    i = 0
		    j = 0
		    while i < len(A) or j < len(B):
			    if j >= len(B) or (i < len(A) and A[i][1] <= B[j][1]):
				    yield A[i]
				    i += 1
			    else:
				    yield B[j]
				    j += 1

	    # Find closest pair recursively; returns all points sorted by y coordinate
	    def recur(L):
		    if len(L) < 2:
			    return L
		    split = len(L)/2
		    splitx = L[split][0]
		    L = list(merge(recur(L[:split]), recur(L[split:])))

		    # Find possible closest pair across split line
		    # Note: this is not quite the same as the algorithm described in class, because
		    # we use the global minimum distance found so far (best[0]), instead of
		    # the best distance found within the recursive calls made by this call to recur().
		    # This change reduces the size of E, speeding up the algorithm a little.
		    #
		    E = [p for p in L if abs(p[0]-splitx) < best[0]]
		    for i in range(len(E)):
			    for j in range(1,8):
				    if i+j < len(E):
					    testpair(E[i],E[i+j])
		    return L
	
	    L.sort()
	    recur(L)
	    return best[1]
            
                                

    def get_distance_between_coords(self, A, B):
        return np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

    def get_nearest(self, POINT, list_of_points):
        nearest = 1000000000
        for x in list_of_points:
            d = self.get_distance_between_coords(POINT, x)
            if d < nearest:
                nearest = d
                winner = x
        return winner

    def get_furthest(self, POINT, list_of_points):
        furthest = 0.0
        for x in list_of_points:
            d = self.get_distance_between_coords(POINT, x)
            if d > furthest:
                furthest = d
                winner = x
        return winner

    def get_distance_from_body_ellipse(self, bodyCentroid, headPoint, POINT):
        perp_to_centroid = self.compute_perpendicular_from_points(bodyCentroid, headPoint)
        perp_x = bodyCentroid[0] + 10.0
        perp_y = perp_to_centroid[0]*(bodyCentroid[0] + 10.0) + perp_to_centroid[1]
        perpPoint = (perp_x, perp_y)
        THETA = self.compute_angle_given_three_points(bodyCentroid, headPoint, perpPoint)
        
        POINT[0] = a*np.cos(THETA)*np.cos(t) - b*np.sin(THETA)*np.sin(t)
        POINT[1] = a*np.sin(THETA)*np.cos(t) + b*np.cos(THETA)*np.sin(t)
        pass   #INCOMPLETE  
        
    def get_centroid_and_head(self, _timestamp):
        centroid = (int(self.bagdf['Bx'].asof(_timestamp)),int(self.bagdf['By'].asof(_timestamp)))#[self.bagdf.Time >= _timestamp].iloc[0]
        head = (int(self.bagdf['Hx'].asof(_timestamp)),int(self.bagdf['Hy'].asof(_timestamp)))
        return centroid, head



    def get_tracking_thresholds(self, _timestamp, _distance, _dTarget):

        if _dTarget <= 20:
            return (75,1,1), (100,2,3), (35,1,1)
        elif _distance <=150:
            return (70,1,1), (102,1,3), (40,1,1)
        elif _distance <=185:
            return (60,1,1), (80,1,3), (30,1,1)
        else:
            return (30,1,1), (67,2,3), (30,1,1)       

    def get_nearest_and_furthest_from_centroid(self, hullset, centroid):
        #PASS A SET OF POINTS DEFINING A SINGLE CONTOUR, IDEALLY OUTPUT FROM cv2.convexHull
        lowest_distance = 1000000
        lowest_coords = (0,0)
        highest_distance = 0
        highest_coords = (0,0)
        for a in hullset:
            b = (a[0][0], a[0][1])
            distance = self.get_distance_between_coords(centroid, b)
            if distance > highest_distance:
                highest_coords = b
                highest_distance = distance
            if distance < lowest_distance:
                lowest_coords = b
                lowest_distance = distance 
        return lowest_coords, highest_coords
        
    def compute_axis_from_points(self, POINT1, POINT2):
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

    def convert_ellipseAngle_to_slope(self, _degs): #OPENCV makes silly angles, where up is 0deg, and right is 90deg.
        degs = float(1.0*_degs + 90.0)
        return float(math.tan(math.radians(degs)))

    def pointOfIntersection(self, SLOPE1, YINT1, SLOPE2, YINT2):
        if float(SLOPE1 - SLOPE2) == 0.0:
            return 
        else:
            px = float(YINT2 - YINT1) / float(SLOPE1 - SLOPE2) 
            py = SLOPE1*px + float(YINT1)
        return (px, py)

    def pointInEllipse(self, x,y,xp,yp,d,D,angle):
        #tests if a point[xp,yp] is within
        #boundaries defined by the ellipse
        #of center[x,y], diameters d D, and tilted at angle

        cosa=math.cos(angle)
        sina=math.sin(angle)
        dd=d/2*d/2
        DD=D/2*D/2

        a =math.pow(cosa*(xp-x)+sina*(yp-y),2)
        b =math.pow(sina*(xp-x)-cosa*(yp-y),2)
        ellipse=(a/dd)+(b/DD)

        if ellipse <= 1:
            return True
        else:
            return False

    def fociOfEllipse(self, x,y,d,D,angle):
        #returns coordinates of foci
        #defined by the ellipse
        #of center[x,y], diameters d D, and tilted at angle

        cosa=math.cos(math.radians(angle-90.0))
        sina=math.sin(math.radians(angle-90.0))
        dd=d/2*d/2
        DD=D/2*D/2

      
        c = np.sqrt(DD-dd)
        slope = self.convert_ellipseAngle_to_slope(angle)
        c_x = cosa*c
        c_y = sina*c
        F1 = ((x+c_x),(y+c_y))
        F2 = ((x-c_x),(y-c_y))
        return (F1, F2)
        
        
    def compute_perpendicular_from_points(self, POINT1, POINT2): #perpendicular line through POINT1
        if  float(float(POINT1[1]) - float(POINT2[1]) ) == 0.0:
            XINT = np.nan
            YINT = POINT1[1]
            SLOPE = 0.0
        else:    
            SLOPE = -1.0*( float(POINT1[0]) - float(POINT2[0])) / ( float(float(POINT1[1]) - float(POINT2[1]) ))
            YINT = float(POINT1[1]) - (float(POINT1[0])*SLOPE)
            if abs(SLOPE) >= 1000000:
                XINT = POINT1[0]
            elif SLOPE == 0.0:
                XINT = np.nan
            else:
                XINT = -1.0*YINT / SLOPE
        return SLOPE, YINT, XINT

    def compute_angle_given_three_points(self, VERTEX, POINT1, POINT2):
        A = np.array(POINT1)
        B = np.array(VERTEX)
        C = np.array(POINT2)
        BA = A - B
        BC = C - B
        s = np.arctan2(*BA)
        e = np.arctan2(*BC)
        return e-s
        
    def check_laterality(self, POINT1, POINT2, SLOPE, YINT, XINT): #TRUE IF TWO POINTS ARE ON THE SAME SIDE OF THE LINE.
        if abs(SLOPE) == np.inf:
            SIGN = (POINT1[0]-XINT)*(POINT2[0]-XINT)  #JUST COMPARE X VALUES TO X-INTERCEPT     
        else:
            SIGN = (SLOPE*POINT1[0] + YINT - POINT1[1])*(SLOPE*POINT2[0] + YINT - POINT2[1])
        
        if SIGN > 0:
            match = 1
        elif SIGN <= 0:
            match = 0
        return match
        
        



if __name__ == "__main__":

    fmf_file = '/media/DBATH_7/150707/R3lexTNT26992L-SS01538-redstim_zoom_20150707_141654/R3lexTNT26992L-SS01538-redstim_zoom_20150707_141654.fmf'
    bag_fn = '/media/DBATH_7/150707/BAGS/rosbagOut_2015-07-07-14-16-53.bag'

    #bagdf = get_data_from_bag(bag_fn)
    #positions = get_positions_from_bag(bag_fn)

    #WIDE_FMF = utilities.match_wide_to_zoom(fmf_file, fmf_file.rsplit('/',2)[0])
    WIDE_FMF = '/media/DBATH_7/150707/R3lexTNT26992L-SS01538-redstim_wide_20150707_141654.fmf'
    targets = target_detector.TargetDetector(WIDE_FMF, fmf_file.rsplit('/',1)[0])
    dtarget = targets.get_dist_to_nearest_target(bag_fn)['dtarget']
    (arena_centre), arena_radius = targets._arena.circ
    wings = WingDetector(fmf_file, bag_fn, dtarget, arena_centre, '/groups/dickson/home/bathd/Desktop/DEBUGGING/' )

     

    times = []
    Lwing = []
    Rwing = []
    import random
    for frame_number in range(8025,8125,1):#random.sample(range(0,11000), 10):
        t, length, width, L, R = wings.detectWings(True, True, frame_number)
        times.append(t)
        Lwing.append(L)
        Rwing.append(R)
        
        
    wingData = DataFrame({'Timestamp':times, 'wing_L':Lwing, 'wing_R':Rwing}, dtype=np.float64)
    wingData.to_pickle('/groups/dickson/home/bathd/Desktop/DEBUGGING/wingdata.pickle')
    print 'Done.'

