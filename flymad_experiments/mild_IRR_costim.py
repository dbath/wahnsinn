#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv

from flymad.constants import LASERS_ALL_OFF, LASER2_ON, LASER1_ON

class Experiment:
    def __init__(self):
        #the IR laser is connected to 'laser1'
        self._ir_laser_conf = rospy.Publisher('/flymad_micro/laser1/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive      
        #configure the lasers
        self._ir_laser_conf.publish(enable=True,   #always enabled, we turn it on/off using /experiment/laser
                                 frequency=5,      
                                 intensity=1.0)    #full power
        

        #Red laser is connected to 'laser2'
        self._red_laser_conf = rospy.Publisher('/flymad_micro/laser2/configuration',
                                            flymad.msg.LaserConfiguration,
                                            latch=True) #latched so message is guarenteed to arrive
        self._red_laser_conf.publish(enable=True, frequency=15,intensity=1.0)

        #ensure the targeter is running so we can control the laser
        rospy.loginfo('waiting for targeter')
        rospy.wait_for_service('/experiment/laser')
        self._laser = rospy.ServiceProxy('/experiment/laser', flymad.srv.LaserState)
        self._laser(LASERS_ALL_OFF)


    def run(self):
        T_WAIT          = 60
        T_IR            = 5
        T_IR_AND_RED    = 10
        T_RED           = 10
        T_WAIT2         = 20

        RED_LASER = LASER2_ON
        IR_LASER  = LASER1_ON

        #the protocol is 10s wait, 5s IR only, 5s IR+red pulse, 5s red constant, 5s no stimulation
        #experiment continues until the node is killed

        rospy.loginfo('running %ss IR only, %ss IR+red, %s red only   NOT TRUE' % (T_IR, T_IR_AND_RED, T_RED))

        try:
            repeat = 0
            rospy.sleep(T_WAIT)
            while ( repeat < 1 ) and not (rospy.is_shutdown()):
                rospy.loginfo('IR only')
                self._laser(IR_LASER)
                rospy.sleep(T_IR)
                #turn on IR and red
                rospy.loginfo('IR + Red pulse')
                self._laser(RED_LASER | IR_LASER)
                rospy.sleep(T_IR_AND_RED)
                #turn off
                rospy.loginfo('Off')
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(T_WAIT2)
                #turn on red
                rospy.loginfo('Red')
                self._laser(RED_LASER)
                rospy.sleep(T_RED)
                #turn off all
                rospy.loginfo('Off')
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(T_WAIT2)
                #turn on IR
                rospy.loginfo('IR  pulse')
                self._laser(IR_LASER)
                rospy.sleep(30)
                #turn off
                rospy.loginfo('Off')
                self._laser(LASERS_ALL_OFF)
                rospy.sleep(T_WAIT2)
                repeat += 1

                print '\a', 'EXPERIMENT FINISHED'
                rospy.sleep(0.5)
                print '\a', '\a', '\a'
                rospy.sleep(0.2)
                print '\a'
                rospy.sleep(0.2)
                print '\a'
                rospy.sleep(0.5)
                print '\a'

        except rospy.ROSInterruptException:
            #the node was cleanly killed
            pass

        self._laser(LASERS_ALL_OFF)

if __name__ == "__main__":
    rospy.init_node('experiment')
    e = Experiment()
    e.run()


