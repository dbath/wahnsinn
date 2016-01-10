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
        self._total_stim = 0

        self._OK_to_initialize = 0
        self._tracking_accuracy = rospy.Subscriber('/flymad/tracked', 
                                            flymad.msg.TrackedObj, self.on_tracked)

        #the position of the currently targeted object is sent by the targeter. In the
        #case of a single fly, the default behaviour of the targeter is to target the
        #one and only fly. If there are multiple flies you neet to instruct the targeter
        #which fly to target. The requires you recieve the raw position of all tracked
        #objects, /flymad/tracked, and decide which one is interesting by sending
        #/flymad/target_object with.


    def on_targeted(self, msg):
        #target flies in the right half of the arena
        if np.sqrt((self.center[0]-msg.fly_x)**2 + (self.center[1]-msg.fly_y)**2) < self.radius:
            rospy.loginfo('targeting fly %s' % msg.obj_id + '\t' +  str(self.stimcount))
            self._laser(LASER0_ON | LASER2_ON)
            self._total_stim += 1
            if self.stimcount >= 1000:
                self._laser(LASER0_ON)
                rospy.sleep(10)
                self.stimcount = 0
            else:
                self.stimcount += 1
                    
        else:
            
            self._laser(LASER0_ON)


    def on_tracked(self, msg):
        foo = abs(msg.state_vec[2])
        bar = abs(msg.state_vec[3])
        if (foo < 1) and (bar <1):
            self._OK_to_initialize = 1
        else:
            self._OK_to_initialize = 0
        return foo

    def run(self):        
        T_WAIT          = 10
        RED_BOUTS        = 5
        T_RED_ON         = 0.5
        T_RED_OFF        = 1.5
        T_RED           = 0
        T_WAIT2         = 300

        RED_LASER = LASER2_ON
        IR_LASER  = LASER1_ON

        #the protocol is 10s wait, 5s IR only, 5s IR+red pulse, 5s red constant, 5s no stimulation
        #experiment continues until the node is killed

        #rospy.loginfo('running %ss IR only, %ss IR+red, %s red only   NOT TRUE' % (T_IR, T_IR_AND_RED, T_RED))
        #rospy.loginfo('running %ss IR then %ss RED' % (((T_IR_ON + T_IR_OFF) * IR_BOUTS), T_RED))
        initialized_target = 0
        try:
            repeat = 0
            rospy.sleep(T_WAIT)


            while ( repeat < 1 ) and not (rospy.is_shutdown()):
                if (repeat == 0):
                    #rospy.loginfo(self._OK_to_initialize)
                    if (initialized_target ==0) and (self._OK_to_initialize == 1):
                        rospy.loginfo( "INITIATING STIMULUS PROTOCOL")
                        initialized_target +=1
                    else:
                        rospy.loginfo("...WAITING FOR ACCURATE TARGETING")
                        continue
                rospy.loginfo('IR only')
                
                for x in range(RED_BOUTS):
                    rospy.loginfo('RED ON \t' + str(x+1))
                    self._laser(RED_LASER | LASER0_ON)
                    rospy.sleep(T_RED_ON)
                    self._laser(LASER0_ON)
                    rospy.loginfo('RED OFF')
                    rospy.sleep(T_RED_OFF)
                #turn off
                rospy.loginfo('Off')
                self._laser(LASER0_ON)
                repeat += 1
                rospy.sleep(5)
                _ = rospy.Subscriber('/targeter/targeted',
                        flymad.msg.TargetedObj,
                        self.on_targeted)
                rospy.loginfo('running')
                while self._total_stim <= 6000:
                    rospy.spin()
                
                rospy.sleep(T_WAIT2)
                
                print '\a', 'EXPERIMENT FINISHED'
                rospy.sleep(0.5)
                print '\a', '\a', '\a'
                rospy.sleep(0.2)
                print '\a'
                rospy.sleep(0.2)
                print '\a'
                rospy.sleep(0.5)
                print '\a'                


                self._laser(LASER0_ON)


        except rospy.ROSInterruptException:
            #the node was cleanly killed
            pass
            

if __name__ == "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stimbouts', type=int, required=False, default=6,
                            help='number of bouts of stimulus') 
    parser.add_argument('--stimlength', type=float, required=False, default=5,
                            help='duration of stimulus in seconds') 
    parser.add_argument('--recoverlength', type=float, required=False, default=5,
                            help='duration of recovery in seconds') 
    parser.add_argument('--center', nargs=2, metavar=('center_x','center_y'),
                        default=(280, 227), help='the (X,Y) location of the arena center')
    parser.add_argument('--radius', type=float,
                        default=33.0, help='the radius of the activation zone')
    args = parser.parse_args()
    
    
    rospy.init_node('experiment')
    e = Experiment(args.center, args.radius)
    e.run()


