#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv
import argparse
import numpy as np


from flymad.constants import LASERS_ALL_OFF, LASER0_ON, LASER1_ON, LASER2_ON

class Experiment:
    def __init__(self, center, radius):
        
        self.center = center
        self.radius = radius
        
        #the red laser is connected to 'laser2'        
        self._red_laser_conf = rospy.Publisher('/flymad_micro/laser2/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive
        #configure the laser
        self._red_laser_conf.publish(enable=True,      #always enabled, we turn it on/off using /experiment/laser
                                 frequency=25.0,      #constant (not pulsed)
                                 intensity=1.0)    #full power
        #Ambient light is connected to 'LASER0'
        self._ambient_conf = rospy.Publisher('/flymad_micro/laser0/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True)
        self._ambient_conf.publish(enable=True, frequency=0, intensity=0.0)
        #ensure the targeter is running so we can control the laser
        rospy.loginfo('waiting for targeter')
        rospy.wait_for_service('/experiment/laser')
        self._laser = rospy.ServiceProxy('/experiment/laser', flymad.srv.LaserState)
        self._laser(LASER0_ON)
        self.stimcount = 0

        #the position of the currently targeted object is sent by the targeter. In the
        #case of a single fly, the default behaviour of the targeter is to target the
        #one and only fly. If there are multiple flies you neet to instruct the targeter
        #which fly to target. The requires you recieve the raw position of all tracked
        #objects, /flymad/tracked, and decide which one is interesting by sending
        #/flymad/target_object with.
        _ = rospy.Subscriber('/targeter/targeted',
                             flymad.msg.TargetedObj,
                             self.on_targeted)

    def on_targeted(self, msg):
        #target flies in the right half of the arena
        if np.sqrt((self.center[0]-msg.fly_x)**2 + (self.center[1]-msg.fly_y)**2) < self.radius:
            rospy.loginfo('targeting fly %s' % msg.obj_id + '\t' +  str(self.stimcount))
            self._laser(LASER0_ON | LASER2_ON)
            
            """if self.stimcount >= 300:
                self._laser(LASER0_ON)
                rospy.sleep(10)
                self.stimcount = 0
            else:
                self.stimcount += 1
            """        
        else:
            
            self._laser(LASER0_ON)

    def run(self):
        rospy.loginfo('running')
        rospy.spin()
        self._laser(LASER0_ON)


            

if __name__ == "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stimbouts', type=int, required=False, default=6,
                            help='number of bouts of stimulus') 
    parser.add_argument('--stimlength', type=float, required=False, default=5,
                            help='duration of stimulus in seconds') 
    parser.add_argument('--recoverlength', type=float, required=False, default=5,
                            help='duration of recovery in seconds') 
    parser.add_argument('--center', nargs=2, metavar=('center_x','center_y'),
                        default=(302, 233), help='the (X,Y) location of the arena center')
    parser.add_argument('--radius', type=float,
                        default=22.0, help='the radius of the activation zone')
    args = parser.parse_args()
    
    
    rospy.init_node('experiment')
    e = Experiment(args.center, args.radius)
    e.run()


