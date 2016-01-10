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
import progressbar
import os
import sh
import shutil

class WingDetector(object):

    def __init__(self, zoomFMF_filepath, bag_filepath, dTarget, arena_centre, tempdir=None ):
        
        
        self.fmf_file = zoomFMF_filepath
        self.fmf = FMF.FlyMovie(self.fmf_file)
        
        self.bag_fn = bag_filepath
        self.bagdf = self.get_data_from_bag(self.bag_fn)
        self.bagdf = self.compute_body_axes(self.bagdf)
        self.positions = self.get_positions_from_bag(self.bag_fn)
        self.positions.loc[self.positions['Px'] == 1000000, 'Px'] = np.nan
        self.positions.loc[self.positions['Py'] == 1000000, 'Py'] = np.nan
        
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
        
        self.DEBUGGING_DIR = self.fmf_file.rsplit('/',1)[0] + '/tracking_cache'
        
        if not os.path.exists(self.DEBUGGING_DIR) == True:
            os.makedirs(self.DEBUGGING_DIR)
        
        self.DEBUGGING_DIR = self.DEBUGGING_DIR + '/'
        self.error_count = 0
        self.ERROR_REPORTING = False
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.previous_head_extended = None
        self.flipped = 0
        self.adjust_tracking_parameters = ((0,0,0),(0,0,0),(0,0,0))
        self.total_errors = 0
       
        self.wingData = DataFrame({'BodyAxis':[],  'leftAngle':[], 'leftWingLength':[], 'Length':[],  'rightAngle':[],'rightWingLength':[],'target_angle_TTM':[], 'target_distance_TTM':[], 'Timestamp':[],'Width':[]}, dtype=np.float64)            

        self.tracking_info = DataFrame({'a_wingAngle_left':[],'a_wingArea_left':[],'b_wingAngle_right':[], 'b_wingArea_right':[], 'c_head_location_x':[],'c_head_location_y':[], 'd_bodyAxis':[], 'e_centroid_x':[], 'e_centroid_y':[], 'f_dTarget_TTM':[], 'g_approachAngle_TTM':[]}, dtype=np.float64)
    
    def execute(self):
    
        total_frames = self.fmf.get_n_frames()
        
        if not self.ERROR_REPORTING:
            progress = self.get_progress_bar("TRACKED", total_frames) 
        else:
            pass
        
        if os.path.exists(self.DEBUGGING_DIR + 'wingdata_cache.pickle'):
            self.wingData = pd.read_pickle(self.DEBUGGING_DIR + 'wingdata_cache.pickle')
            self.wingData.columns= ['BodyAxis','leftAngle','leftWingLength','Length','rightAngle','rightWingLength','target_angle_TTM',
                                     'target_distance_TTM','Timestamp','Width']
            startframe = self.wingData.index[-1]
            print self.fmf_file.split('/')[-1], ': beginning from cache at: ', startframe
        else:
            startframe = 0
            
        for frame_number in range(startframe,total_frames,1):
            if self.ERROR_REPORTING:
                progress = self.get_progress_bar("ERROR_RATE", 2*frame_number+1) 
                progress.update(self.total_errors+1)   
            else:
                progress.update(frame_number) 
            self.ERROR_DETECTED= False
            self.error_count = 0
            self.adjust_tracking_parameters = ((0,0,0),(0,0,0),(0,0,0))
            self.detectWings(self.saveImage, False, frame_number)  #MAKE FIRST OPTION TRUE TO SAVE TRACKING MOVIES.
        print self.fmf_file.split('/')[-1], 100.0*self.total_errors/total_frames, '% error rate'
        return



    def make_movie(self,imagepath,filename,mp4fps):

        #write x264 mp4
        tmpmov = "%s/movie.y4m" % imagepath

        sh.mplayer("mf://%s/*.png" % imagepath,
                   "-mf", "fps=%d" % mp4fps,
                   "-vo", "yuv4mpeg:file=%s" % tmpmov,
                   "-ao", "null",
                   "-nosound", "-noframedrop", "-benchmark", "-nolirc"
        )

        sh.x264("--output=%s" % filename,
                "%s" % tmpmov,
        )


        try:
            os.unlink(tmpmov)
            shutil.rmtree(self._tempdir)
        except OSError:
            pass

        
    def get_progress_bar(self, name, maxval):
        widgets = ["%s: " % name, progressbar.Percentage(),
                   progressbar.Bar(), progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets,maxval=maxval).start()
        return pbar


    def get_wingAngle(self, frame_number):
        t, L, R = self.detectWings(frame_number)
        return t, L, R
    
    def devignette(self, frame):
        
        if int(self.fmf_file.rsplit('_')[-2]) >= 151006:
            V_coeff =[ 0.608421,0.000660594,0.00071838,
                       -6.83654e-07,2.29008e-07,-6.11814e-07,
                       -8.79999e-11,-1.63231e-10,-2.10072e-11,-2.10298e-10]
        else:  
            
            V_coeff = [  5.198890393267561e-01,
                         1.217460251226269e-03,
                         1.189236244172212e-03,
                        -1.476571361684494e-06,
                        -6.157281314884152e-07,
                        -1.611555274365404e-06,
                         2.521929214022170e-10,
                         4.392272775279915e-10,
                         2.268726532499034e-10,
                         4.244172315090120e-10]



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
            try:
                px.append(msg.points[0].x)
                py.append(msg.points[0].y)
            except:
                px.append(1000000)
                py.append(1000000)
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
        timestring = "%.2f" % (pd.to_datetime(timestamp) - pd.to_datetime(0)).total_seconds()
        
        # COMPUTER VISION:
        frame = self.devignette(frame)
        im = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #must be uint8 array
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((5,5),np.uint8)


        Px = self.positions.Px.asof(timestamp_FMT)    #SILLY HACK FOR 'MISMATCHING' INDICES. STUPID PANDAS.
        Py = self.positions.Py.asof(timestamp_FMT)
        
        if Px == np.nan or Py == np.nan:
            if self.saveImage == True:
                imcopy = im.copy()
                cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy) 
            self.wingData.loc[framenumber] = [np.nan, np.nan, np.nan,  np.nan,   np.nan, np.nan, timestamp, np.nan]
            return np.nan, np.nan, np.nan,  np.nan,   np.nan, np.nan, timestamp, np.nan
        
        distance = self.get_distance_between_coords((Px,Py), self.arena_centre)
        targ_dist = self.dTarget.asof(timestamp_FMT)        


        #FLY FEATURES DERIVED FROM BAG FILE:
        try:
            centroid, head = self.get_centroid_and_head(timestamp_FMT)
            backPoint = tuple(sum(y) / len(y) for y in zip(centroid, head))
            headLine = self.compute_perpendicular_from_points(head, centroid)
            axisLine = self.compute_axis_from_points(head, centroid)
            bodyThresh, wingThresh, ellThresh = self.get_tracking_thresholds(timestamp_FMT, distance, targ_dist)
            BagData = True
        except:
            centroid, head = (0,0),(0,0)
            bodyThresh, wingThresh, ellThresh = self.get_tracking_thresholds(timestamp_FMT, distance, targ_dist) 
            BagData = False          


        #FIT ELLIPSE TO BODY:
        ret2, body = cv2.threshold(imgray, ellThresh[0], 255, cv2.THRESH_BINARY)
        #ellipseFitter = cv2.dilate(body, kernel, iterations=ellThresh[1])
        ellipseFitter = cv2.erode(body, kernel, iterations=ellThresh[2])
        contourImage = ellipseFitter.copy()
        bodyCont, hierarchy1 = cv2.findContours(contourImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        

        bodyEllipse=None
        bodyContour=None
        
        if BagData:
            for cnt in bodyCont:
                if cv2.contourArea(cnt) <=900000:
                    if cv2.contourArea(cnt) >= 7000:
                        ellipse= cv2.fitEllipse(cnt)
                        if self.pointInEllipse(centroid[0],centroid[1],ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2]):
                            bodyEllipse = ellipse
                            bodyContour=cnt
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
                        bodyContour=cnt
        
        if bodyEllipse == None:
            #print "ERROR: cannot detect body ellipse in frame: ", framenumber
            
            imcopy = im.copy()
            cv2.putText(imcopy, "ERROR", (480,530), self.font, 1, (255,255,255), 3)
            self.wingData.loc[framenumber] = self.wingData.loc[framenumber-1]#[np.nan, np.nan, np.nan,  np.nan, np.nan. np.nan]

            if self.saveImage == True:
                cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy)  
            return timestamp, np.nan, np.nan, np.nan, np.nan, np.nan             
        
        
        
        (f1, f2) = self.fociOfEllipse(bodyEllipse[0][0],bodyEllipse[0][1],bodyEllipse[1][0],bodyEllipse[1][1],bodyEllipse[2])

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
        midline = self.compute_perpendicular_from_points(centroid, head)
        tailLine = self.compute_perpendicular_from_points(tail, centroid)
    
    

        
        #FLIP BODY AXIS BASED ON PREVIOUS FRAMES
        
        
        
        ########################################################################################
        
        try:
            if not self.previous_head_extended == None:
            
                #print framenumber, ': ', self.wingData.ix[framenumber-1].BodyAxis, bodyEllipse[2], np.cos(np.radians(self.wingData.ix[framenumber-1].BodyAxis - bodyEllipse[2])), '\t', self.flipped
            
                if not self.check_laterality(self.previous_head_extended, self.extend_vector(centroid,head), midline[0], midline[1], midline[2]):
                    #debugging = True
                    if self.flipped == 100:
                        for x in range(-101,1):
                            self.previous_head_extended = None
                            self.detectWings(True, True, framenumber+x)
                        self.flipped = 0
                        return
                    head, tail = tail, head
                    headLine, tailLine = tailLine, headLine
                    backPoint = tuple(sum(y) / len(y) for y in zip(centroid, head))
                    backPoint = tuple(sum(y) / len(y) for y in zip(centroid, backPoint))

                    self.flipped += 1
                else: self.flipped = 0
        except:
            pass
            #print framenumber, ": Unable to assess body orientation."
        self.previous_head_extended = self.extend_vector(centroid,head)
        
        
        body_length = self.get_distance_between_coords(head,tail)
        abd_length =  self.get_distance_between_coords(backPoint, tail)        
        body_angle = self.angle_from_vertical(tail, head)


        if body_length >= 425:
            imcopy = im.copy()
            cv2.putText(imcopy, "ERROR", (480,530), self.font, 1, (255,255,255), 3)
            if self.saveImage == True:
                cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy)
            self.wingData.loc[framenumber] = [np.nan, np.nan, np.nan,  np.nan,   np.nan, np.nan, np.nan, np.nan, timestamp, np.nan]
            return np.nan, np.nan, np.nan,  np.nan,   np.nan, np.nan, timestamp, np.nan

        WIDTH = bodyEllipse[1][0]

        ########################### TARGET TOUCH #############################################
        
        if (bodyContour != None) & (targ_dist <= 40.0):
            imcopy = imgray.copy()
            bodymask = imcopy / imcopy - 1.0  #zeros, with dimensions of image.
            cv2.fillPoly(bodymask, [bodyContour],(255-imcopy.max()))
            bodymask = cv2.dilate(bodymask, kernel, iterations=4)
            imcopy = (imcopy + bodymask).astype(np.uint8)
            ret, trunk = cv2.threshold(imcopy, 80, 90, cv2.THRESH_TRUNC)
            targetContour, target_distance_TTM, approach_angle_TTM = self.get_targets(trunk, head, centroid, body_angle)
        else:
            targetContour, target_distance_TTM, approach_angle_TTM = [], np.nan, np.nan




        ############################# DEFINE WINGS #########################################################

        wingTips, wholeWings, wingArea = [],[],[]

        wingTips, wholeWings, wingArea = self.get_candidate_wings(imgray, kernel, headLine, centroid, backPoint, body_length, abd_length, axisLine, wingTips, wholeWings, wingArea, timestamp_FMT, distance, targ_dist, ((0,0,0),(0,0,0),(0,0,0)))
        wingTips, wholeWings, wingArea = self.get_candidate_wings(imgray, kernel, headLine, centroid, backPoint, body_length, abd_length,axisLine, wingTips, wholeWings, wingArea,timestamp_FMT, distance, targ_dist, ((-10,0,0),(-10,0,0),(0,0,0)))
        wingTips, wholeWings, wingArea = self.get_candidate_wings(imgray, kernel, headLine, centroid, backPoint, body_length, abd_length,axisLine, wingTips, wholeWings, wingArea,timestamp_FMT, distance, targ_dist, ((10,0,0),(10,0,0),(0,0,0)))
        wingTips, wholeWings, wingArea = self.get_candidate_wings(imgray, kernel, headLine, centroid, backPoint, body_length, abd_length,axisLine, wingTips, wholeWings, wingArea,timestamp_FMT, distance, targ_dist, ((10,1,1),(10,0,1),(0,0,0)))
        wingTips, wholeWings, wingArea = self.get_candidate_wings(imgray, kernel, headLine, centroid, backPoint, body_length, abd_length,axisLine, wingTips, wholeWings, wingArea,timestamp_FMT, distance, targ_dist, ((-10,1,1),(-10,0,1),(0,0,0)))

        polynomial = np.poly1d([  1.47283629e-09,   7.10509872e-06,  -1.29235190e-02])

        wingSets = pd.DataFrame({'Tips':wingTips, 'Shape':wholeWings, 'Area':wingArea, 'Theta':np.empty(len(wingTips)).fill(0)})
        wingSets['Theta'] = np.nan
        wingSets['Side'] = np.nan
        wingSets['Length'] = np.nan
        
        #wingSets.to_pickle('/groups/dickson/home/bathd/Desktop/wingsets.pickle')
        
        for x in np.arange(len(wingSets)):
            wingSets.loc[x,'Theta'] = self.compute_angle_given_three_points(backPoint, wingSets.loc[x,'Tips'], centroid)
            wingSets.loc[x,'Length'] = self.get_distance_between_coords(backPoint, wingSets.loc[x,'Tips'])
        wingSets.loc[wingSets['Theta'] >=np.pi,'Theta'] -= 2.0*np.pi
        wingSets.loc[wingSets['Theta'] <=-1.0*np.pi,'Theta'] += 2.0*np.pi
        wingSets.loc[wingSets['Theta'] < 0.0, 'Side'] = 'Right'
        wingSets.loc[wingSets['Theta'] >= 0.0, 'Side'] = 'Left'
        wingSets.loc[wingSets['Side'] == 'Right','Theta'] *= -1.0
        wingSets['polydif'] = (wingSets['Theta'] - polynomial(wingSets['Area'])) 
                        
        wingSets = wingSets[(wingSets['Length'] >=235) & (wingSets['Length'] < 350)]
        wingSets = wingSets[(wingSets['Theta'] <= (np.pi)/2.0 )]
        wingSets = wingSets[(wingSets['polydif'] >= -0.4) & (wingSets['polydif'] <= 0.6)]
        wingSets = wingSets[(wingSets['Area'] >= 2000) & (wingSets['Area'] <= 35000)]

        
        try:
            #leftWing = wingSets.ix[wingSets[wingSets['Side']=='Left']['polydif'].abs().idxmin()]
            leftWing = wingSets.ix[wingSets[wingSets['Side']=='Left']['Area'].abs().idxmax()]
        except:
            leftWing = wingSets[0:0]
            leftWing.ix[0] = np.nan
            leftWing.set_value(0,'Tips',tuple(tail))
            leftWing.set_value(0,'Shape',[[0,0]])
            leftWing = leftWing.ix[0]
            self.total_errors += 1
        try:
            #rightWing = wingSets.ix[wingSets[wingSets['Side']=='Right']['polydif'].abs().idxmin()]
            rightWing = wingSets.ix[wingSets[wingSets['Side']=='Right']['Area'].abs().idxmax()]
        except:
            rightWing = wingSets[0:0]
            rightWing.ix[0] = np.nan
            rightWing.set_value(0,'Tips',tuple(tail))
            rightWing.set_value(0,'Shape',[[0,0]])
            rightWing = rightWing.ix[0]
            self.total_errors +=1


        if saveImage == True:
            imcopy = im.copy()
            try:
                cv2.drawContours(imcopy,[leftWing.Shape],0,(255,0,0),1)
            except: 
                pass
            try:
                cv2.drawContours(imcopy,[rightWing.Shape],0,(0,255,255),1)
            except: 
                pass
            try:
                cv2.drawContours(imcopy,[targetContour],0,(255,128,128),6)
            except: 
                pass
            cv2.line(imcopy, (int(head[0]),int(head[1])), (int(tail[0]),int(tail[1])), (255,255,255), 1)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(leftWing.Tips[0]),int(leftWing.Tips[1])), (20,20,255),2)
            cv2.line(imcopy, (int(backPoint[0]),int(backPoint[1])), (int(rightWing.Tips[0]),int(rightWing.Tips[1])), (20,255,20),2)
            cv2.circle(imcopy, (int(head[0]),int(head[1])), 3, (255,255,255), -1)
            cv2.circle(imcopy, (int(backPoint[0]),int(backPoint[1])), 5, (255,255,255), -1)
            #cv2.circle(imcopy, (int(centroid[0]),int(centroid[1])), 3, (255,0,255), -1)
            cv2.putText(imcopy, str(np.around(np.degrees(leftWing.Theta), 2))+ 'deg', (10,25), self.font, 1, (20,20,255), 3)
            cv2.putText(imcopy, str(np.around(np.degrees(rightWing.Theta), 2))+ 'deg', (450, 25), self.font, 1, (20,255,20), 3)
            cv2.putText(imcopy, str(framenumber), (850, 25), self.font, 1, (255,255,255), 3)
            cv2.putText(imcopy, str(np.around(target_distance_TTM, 2)) + 'mm', (10,950), self.font, 1, (100,255,255), 3)
            cv2.putText(imcopy, str(np.around(approach_angle_TTM, 2)) + 'deg', (450, 950), self.font, 1, (100,255,255), 3)
            #cv2.putText(imcopy, timestring, (850, 950), self.font, 1, (255,255,255), 3)
            cv2.imwrite(self._tempdir+'_tmp%05d.png'%(framenumber), imcopy) 
        cv2.destroyAllWindows()

        
        #print framenumber,  "\tL: ", ("%.2f" % np.degrees(leftWingAngle)), ("%.2f" % leftWingLength), '\tR: ', ("%.2f" % (-1.0*np.degrees(rightWingAngle))), ("%.2f" % rightWingLength), '\t',("%.2f" % distance), '\t', str(self.dTarget.asof(timestamp_FMT)), '\t', self.flipped
        
        self.wingData.loc[framenumber] = [body_angle, leftWing.Theta, leftWing.Length,  body_length,
                                          rightWing.Theta, rightWing.Length, 
                                          approach_angle_TTM, target_distance_TTM, timestamp, WIDTH]
        self.tracking_info.loc[framenumber] = [leftWing.Theta, leftWing.Area, rightWing.Theta, 
                                               rightWing.Area, head[0], head[1], body_angle, centroid[0], centroid[1], target_distance_TTM, approach_angle_TTM]


        if framenumber % 100 == 0:
            self.wingData.to_pickle(self.DEBUGGING_DIR + 'wingdata_cache.pickle')

        return body_angle, leftWing.Length, leftWing.Theta, body_length, rightWing.Length,  rightWing.Theta, timestamp, WIDTH

    def get_targets(self, fly_erased_img, headpoint, centroidpoint, _bodyAxis):

        kernel = np.ones((5,5),np.uint8)
        _, mask = cv2.threshold(fly_erased_img, 70, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, kernel, iterations=1)
        contourImage = mask.copy()
        contourImage = np.pad(contourImage,((2,2),(2,2)), mode='maximum')
        contours, hierarchy1 = cv2.findContours(contourImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy1[0]
        
        for x in hierarchy:
            if x[3] <0:
                parent = x
        
        #headpoint = (int(track.loc[framenumber, 'c_head_location_x']), int(track.loc[framenumber, 'c_head_location_y']))
        candidateTargets = []
        
        for component in zip(contours, hierarchy):
            c = component[0]
            h = component[1]
            centroidCheck = cv2.pointPolygonTest(c,centroidpoint,True)
            if centroidCheck <=0:
                if np.array_equal(hierarchy[h[3]], parent) :  #is in outer hierarchy (parent is edge.)  
                    if h[2] > 0:   # has child (targets have inner and outer edge)
                        if (cv2.contourArea(c) <= 500000) & (cv2.contourArea(c) >= 20000):
                            ellipse = cv2.fitEllipse(c)
                            if not self.pointInEllipse(centroidpoint[0],centroidpoint[1],ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2]):
                                candidateTargets.append(c)
            
            areas = []
            if len(candidateTargets) >0:
                for T in range(len(candidateTargets)): 
                    areas.append(cv2.contourArea(candidateTargets[T]))
            
                TARGET = cv2.convexHull(candidateTargets[areas.index(max(areas))]             )
                M = cv2.moments(TARGET)
                targCentre = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
                    
                distance = -1.0*cv2.pointPolygonTest(TARGET,headpoint,True) / 135.5 # based on 135.5 pixels per mm
                angle= self.angle_from_vertical(headpoint, targCentre)
                approachAngle= angle - _bodyAxis #track.loc[framenumber, 'd_bodyAxis']
                if approachAngle < 0:
                    approachAngle *= -1.0
                if approachAngle >=180.0:
                    approachAngle -= 180.0
            else:   
                distance = np.nan
                approachAngle = np.nan
                TARGET = None
        return TARGET, distance, approachAngle
    
    def get_candidate_wings(self, imgray, kernel, headLine, centroid, backPoint, body_length, abd_length, axisLine, wingTips, wholeWings, wingArea,timestamp_FMT, distance, targ_dist, parameter_adjustment):
        
        self.adjust_tracking_parameters = parameter_adjustment

        bodyThresh, wingThresh, ellThresh = self.get_tracking_thresholds(timestamp_FMT, distance, targ_dist)
        
        #DEFINE bodyNotWings AS BODY PORTION PLUS LEGS ETC, USEFUL FOR FINDING WINGS.
        ret1, bodyNotWings = cv2.threshold(imgray, bodyThresh[0],255,cv2.THRESH_BINARY)
        bodyNotWings = cv2.dilate(bodyNotWings, kernel, iterations=bodyThresh[1])
        bodyNotWings = cv2.erode(bodyNotWings, kernel, iterations=bodyThresh[2])

        
        #DEFINE wings AS WINGS AND TARGETS BUT NOT BODY.
        ret2, wings = cv2.threshold(imgray, wingThresh[0],1,cv2.THRESH_BINARY_INV)
        test = wings*bodyNotWings
        dilated = cv2.erode(test, kernel, iterations=wingThresh[2])
        eroded = cv2.dilate(dilated, kernel, iterations=wingThresh[1])
        dilatedCopy = eroded.copy()
        
        wingCont, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        
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
                    pointSetTARGET = []
                    for x in c:
                        if self.check_laterality((x[0][0], x[0][1]), centroid, headLine[0], headLine[1], headLine[2]):
                            if self.check_laterality((x[0][0], x[0][1]), checkSpot, axisLine[0], axisLine[1], axisLine[2]):
                                pointSet1.append(x.tolist())
                            else:
                                pointSet2.append(x.tolist())
                        else:
                            if targ_dist <=20.0:
                                pointSetTARGET.append(x.tolist())
                    pointSet1 = np.array(pointSet1).reshape((-1,1,2)).astype(np.int32)
                    pointSet2 = np.array(pointSet2).reshape((-1,1,2)).astype(np.int32)
                    pointSetTARGET = np.array(pointSetTARGET).reshape((-1,1,2)).astype(np.int32)
                    if (len(pointSet1) > 0):
                        if cv2.contourArea(pointSet1) >=(2500/(wingThresh[2]+1)):
                            near, far = self.get_nearest_and_furthest_from_centroid(pointSet1, centroid)
                            if self.get_distance_between_coords(near, centroid) <= 150:
                                winglength = self.get_distance_between_coords(far, backPoint)
                                if (winglength <= 1.5*(body_length)) and (winglength >= abd_length):
                                    wingTips.append(far)
                                    wholeWings.append(cv2.convexHull(pointSet1))
                                    wingArea.append(cv2.contourArea(cv2.convexHull(pointSet1)))
                    if (len(pointSet2) > 0):
                        if cv2.contourArea(pointSet2) >=(2500/(wingThresh[2]+1)):
                            near, far = self.get_nearest_and_furthest_from_centroid(pointSet2, centroid)
                            if self.get_distance_between_coords(near, centroid) <= 150:
                                winglength = self.get_distance_between_coords(far, backPoint)
                                if (winglength <= 1.5*(body_length)) and (winglength >= abd_length):
                                    wingTips.append(far)
                                    wholeWings.append(cv2.convexHull(pointSet2))
                                    wingArea.append(cv2.contourArea(cv2.convexHull(pointSet2)))
        return wingTips, wholeWings, wingArea
    





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

        if _dTarget <= 4:
            vals = (80,1,1), (113,1,2), (35,1,1)
        elif _distance <=120:
            vals =  (70,1,1), (110,1,2), (40,1,1)
        elif _distance <=150:
            vals =  (70,1,1), (95,1,2), (40,1,1)
        elif _distance <=185:
            vals = (60,1,1), (85,1,2), (30,1,1) #(60,1,1), (80,1,2), (30,1,1)
        else:
            vals = (55,1,2), (83,1,2), (35,1,1) #(50,1,1), (79,1,2), (35,1,1)   
            
        foo = self.add_nested_tuples(vals, self.adjust_tracking_parameters)
        
        return foo   

    def add_nested_tuples(self, set1, set2):
        return tuple(map(lambda x, y: tuple(map(lambda w,z: w+z, x,y)), set1, set2))

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
    
        
    def extend_vector(self, BACKPOINT, FRONTPOINT):
        delta_x, delta_y = (FRONTPOINT[0]-BACKPOINT[0]), (FRONTPOINT[1] - BACKPOINT[1])
        new_x = FRONTPOINT[0] + delta_x/abs(delta_x)*1000
        new_y = FRONTPOINT[1] + delta_y/abs(delta_y)*1000
        return (new_x, new_y)

    def angle_from_vertical(self, point1, point2):
        """
        RETURNS A VALUE IN DEGREES BETWEEN 0 AND 360, WHERE 0 AND 360 ARE NORTH ORIENTATION.
        """
        x = point1[0] - point2[0]
        y = point1[1] - point2[1]
        return 180.0 + math.atan2(x,y)*180.0/np.pi



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
    wings = WingDetector(fmf_file, bag_fn, dtarget, arena_centre, '/media/DBATH_7/150707/FINALS/' )


    for frame_number in range(000,10000,1):#random.sample(range(0,11000), 10):
        wings.detectWings(True, False, frame_number)

    wings.wingData.to_pickle('/media/DBATH_7/150707/DEBUGGING/wingdata.pickle')
    #print 'Done.'

