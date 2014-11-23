#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import std_msgs.msg
import flymad.msg
import flymad.srv

from flymad.constants import LASERS_ALL_OFF, LASER1_ON, LASER2_ON

_time_on = 5

class Experiment:
    def __init__(self):
        self._time_delay = float(rospy.get_param('~time_delay', 10))  #TIME BEFORE FIRST 'LASER ON' BOUT
        self._time_on = float(rospy.get_param('~time_on', 10)) #DURATION OF 'LASER ON' BOUTS
        self._time_off = float(rospy.get_param('~time_off', 30))  #INTERVAL BETWEEN BOUTS
        self._repeats = int(rospy.get_param('~repeats', 1))  #NUMBER OF BOUTS


        self._laser1_conf = rospy.Publisher('/flymad_micro/laser1/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True)
        self._laser2_conf = rospy.Publisher('/flymad_micro/laser2/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True)

        #the red laser is pulsed at 1 hz
        self._laser2_conf.publish(enable=True,      #we turn and off the lasers using the /experiment/laser message
                                  frequency=30.0,
                                  intensity=1.0)
        #the IR laser is at 0.2hz
        self._laser1_conf.publish(enable=True,
                                  frequency=10.0,
                                  intensity=1.0)


        print "WAITING FOR TARGETER"
        rospy.wait_for_service('/experiment/laser')
        self._laser = rospy.ServiceProxy('/experiment/laser', flymad.srv.LaserState)

        self._laser(LASERS_ALL_OFF)


    def run(self):
        rospy.loginfo('running experiment')
        try:
            repeat = 0
            #rospy.sleep(self._time_delay)
            while (repeat < self._repeats) and (not rospy.is_shutdown()):
                print   'Laser OFF '
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(5)
                redcount = 0
                while (redcount < 10) and (not rospy.is_shutdown()):                
                    print   'RED Laser ON  '
                    self._laser(LASER2_ON)
                    rospy.sleep(0.5)
                    self._laser(LASERS_ALL_OFF)
                    rospy.sleep(9.5)
                    redcount += 1
                self._laser(LASER1_ON)
                print 'IR ON'
                rospy.sleep(10)
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(5)
                redcount = 0
                while (redcount < 10) and (not rospy.is_shutdown()):                
                    print   'RED Laser ON  '
                    self._laser(LASER2_ON)
                    rospy.sleep(2)
                    self._laser(LASERS_ALL_OFF)
                    rospy.sleep(5)
                    redcount += 1
                #self._laser(LASER1_ON | LASER2_ON)
                print 'ALL OFF'
                self._laser(LASERS_ALL_OFF)
                repeat += 1
        except rospy.ROSInterruptException:
            #the node was cleanly killed
            pass
        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


