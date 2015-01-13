
import utilities.utilities as utilities
import flymad_jaaba.target_detector as target_detector

import matplotlib.pyplot as plt


fmf_file = '/media/DBATH_6/150108/DB201-GP-costim_wide_20150108_185354.fmf'

bagdir = '/media/DBATH_6/150108/BAGS'


tempdir = '/groups/dickson/home/bathd/Desktop/'


bag_file = utilities.match_fmf_and_bag(fmf_file, bagdir)

targs = target_detector.TargetDetector(fmf_file, tempdir)


positions = utilities.get_positions_from_bag(bag_file)
distances = targs.get_dist_to_nearest_target(bag_file)


plt.plot(distances)
plt.show()

plt.plot(positions.fly_x, positions.fly_y, alpha=distances.values)
plt.show()
