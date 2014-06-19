#!/usr/bin/env python

import time
import os.path
import glob
import tempfile
import shutil


if __name__ == "__main__":
    BAG_DATE_FMT = "rosbagOut_%Y-%m-%d-%H-%M-%S.bag"
    MP4_DATE_FMT = "%Y%m%d_%H%M%S.mp4"

    destdir = os.path.expanduser("/groups/dickson/home/bathd/Desktop/OUTPUT/")
    inputmp4 = os.path.expanduser("/groups/dickson/home/bathd/Desktop/DROP") 
    inputbags = os.path.expanduser("/groups/dickson/home/bathd/Desktop/DROP")

    mp4s = glob.glob(inputmp4+"/*.mp4")
    bags = glob.glob(inputbags+"/*.bag")


    for bag in bags:
        print 'processing bag file', bag
        btime = time.strptime(os.path.basename(bag), BAG_DATE_FMT)
        matching = {}
        for mp4 in mp4s:
            try:
                mp4fn = os.path.basename(mp4)
                genotype,datestr = mp4fn.split("_",1)
            except ValueError:
                print "invalid mp4name", mp4
                continue

            try:
                mp4time = time.strptime(datestr, MP4_DATE_FMT)
            except ValueError:
                print "invalid mp4fname", mp4
                continue

            dt = abs(time.mktime(mp4time) - time.mktime(btime))

            if (dt < 20):
                matching["genotype"] = genotype
                matching["date"] = btime

        if matching:
            tmpdir = tempfile.mkdtemp()
            destfn = os.path.join(destdir,"%s_%s.bag" % (matching["genotype"], 
                                           time.strftime("%Y%m%d_%H%M%S",matching["date"]))
            )

            print "making",destfn
            try:
                os.rename(bag, destfn)
            except Exception, e:
                print "ERROR making movie: %s\n%s" % (bag,e)
                pass

            shutil.rmtree(tmpdir)


