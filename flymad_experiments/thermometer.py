#!/usr/bin/env python
import roslib
roslib.load_manifest('flymad')

import rospy
import flymad.msg
import flymad.srv

from flymad.constants import LASERS_ALL_OFF, LASER2_ON, LASER1_ON


import std_msgs.msg
import sys
import time
import numpy as np
from std_msgs.msg import String

import Agilent.Devices



class Temperatures:
    def __init__(self, offset):

        self._tempTC = Agilent.Devices.AT34410A('192.168.100.2', debug=True)
        self._Laser1 = rospy.Subscriber('/flymad_micro/laser1/current', 
                                            std_msgs.msg.Float32, self.callback_current1)

        self._Laser2 = rospy.Subscriber('/flymad_micro/laser2/current', 
                                            std_msgs.msg.Float32, self.callback_current2)                            
        self._OFFSET = float(offset)

    def callback_current1(self, msg):  #lazy danno
        self._laser1_current = msg.data
        
    def callback_current2(self, msg):  #lazy danno
        self._laser2_current = msg.data
        




    def getTCvoltage(self):
        # returns Volts
        return float(self._tempTC.SCPI_query_cmd('MEAS:VOLT:DC? 100mV'))

    def voltage2temperature(self,voltage):
        v = voltage*1000. +0.15

        C = [ 0.0,
              2.5928e1,
             -7.602961e-1,
              4.637791e-2,
             -2.165394e-3,
              6.048144e-5,
             -7.293422e-7,
              0.0 ]

        # v (mV) --> T (degC)
        return (C[0] +
                C[1]*(v) +
                C[2]*(v**2) +
                C[3]*(v**3) +
                C[4]*(v**4) +
                C[5]*(v**5) +
                C[6]*(v**6) +
                C[7]*(v**7) ) + self._OFFSET
                                              
    def take_single_measurement(self, f):
        now = time.time()
        v1 = self.getTCvoltage()
        t1 = self.voltage2temperature(v1)
        l1 = self._laser1_current
        l2 = self._laser2_current
        print "time: %fs,\tvoltage: %fV,\ttemperature: %fC, Laser1: %fmA, Laser2: %fmA" % (now, v1, t1, l1, l2)
        f.write(str(now) + '\t' + str(v1) + '\t' + str(t1) +'\t' + str(l1) + '\t' + str(l2) + '\n')
        f.flush()
        pub.publish(str(t1))


if __name__ == '__main__':

    import os


    if len(sys.argv) != 3:
        print 'call with output filename & offset value to zero (degrees celsius)'
        exit()

    fn = sys.argv[1]

    
    temps = Temperatures(sys.argv[2])
      
    """
    if os.path.exists(fn):
        print 'overriding file! change filename'
        exit()
    """

    try:
        with open(fn, 'w') as out:
            out.write('# time (s), ThermocoupleVoltage (V), temperature (deg C), Laser1 (A), Laser2 (A) \n')
            pub = rospy.Publisher('temperature', String)
            rospy.init_node('temperature_node')           
            while True:

                temps.take_single_measurement(out)
    except:
        print "Error"
        raise
    

    




