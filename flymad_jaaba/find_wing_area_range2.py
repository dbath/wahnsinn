import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



datadf = pd.DataFrame({'angle':[], 'area':[]})

flies = 0
for x in glob.glob('/nearline/dickson/bathd/FlyMAD_data_archive/P1_adt2_sequential_activation/150513/*/tracking_info.pickle'):
    df = pd.read_pickle(x)
    lefty = df[df.a_wingAngle_left.notnull()]
    righty = df[df.b_wingAngle_right.notnull()]
    left = pd.DataFrame({'angle':lefty.a_wingAngle_left, 'area':lefty.a_wingArea_left})
    right = pd.DataFrame({'angle':-1.0*righty.b_wingAngle_right, 'area':righty.b_wingArea_right})
    datadf = pd.concat([datadf, left, right])
    flies +=1


    
clipped = datadf[(datadf.angle <1.309) & (datadf.area < 30000)]


coefficients1 = np.polyfit(clipped.area.values, clipped.angle.values, 2)

polynomial1 = np.poly1d(coefficients1)
"""
upper = clipped[(clipped.area > polynomial1(clipped.angle)) & (clipped.angle > 0.5)]

lower = clipped[clipped.angle <=0.5]

df = pd.concat([upper, lower], axis=0)
"""
#df = clipped[(clipped.angle > polynomial1(clipped.area))]
df = clipped.copy()

coefficients2 = np.polyfit(df.area.values, df.angle.values, 2)

polynomial2 = np.poly1d(coefficients2)

xs = np.linspace(0, datadf.area.max(), 1000)
ys = polynomial2(xs)

datadf['polydif'] = polynomial2(datadf['area']) - datadf['angle']

print coefficients2

plt.figure()
plt.subplot(311)
plt.hist2d(datadf.area.values, datadf.angle.values, bins=40, norm=LogNorm())
plt.colorbar()
plt.plot(xs, ys, 'white', linewidth=3)
plt.ylabel('Angle (rad)')
plt.xlabel('Wing area (pixels)')

plt.subplot(323)
plt.hist(datadf.area.values, bins=np.arange(0,30000,300))
plt.xlabel('area')
plt.ylabel('Frequency')
plt.subplot(324)
plt.hist(datadf.polydif.values, bins=np.arange(-1,1,0.02))
plt.xlabel('Difference, wing angle vs polynomial2(area)')
plt.ylabel('Frequency')
plt.text(0.95, 1000, str(flies) + ' flies\n' + str(len(datadf)) + ' wings',
         horizontalalignment='right', verticalalignment='bottom', color='black') 

plt.subplot(313)
plt.hist2d(datadf.area.values, datadf.polydif.values, bins=100, norm=LogNorm())
plt.colorbar()
plt.xlabel('area')
plt.ylabel('Difference, wing angle vs polynomial2(area)')

plt.show()
