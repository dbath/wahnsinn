import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



datadf = pd.DataFrame({'angle':[], 'area':[]})

flies = 0
for x in glob.glob('/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/*/tracking_info.pickle'):
    df = pd.read_pickle(x)
    lefty = df[df.a_wingAngle_left.notnull()]
    righty = df[df.b_wingAngle_right.notnull()]
    left = pd.DataFrame({'angle':abs(lefty.a_wingAngle_left), 'area':lefty.a_wingArea_left})
    right = pd.DataFrame({'angle':abs(righty.b_wingAngle_right), 'area':righty.b_wingArea_right})
    datadf = pd.concat([datadf, left, right])
    flies +=1


    
clipped = datadf[(datadf.angle <1.609) & (datadf.area < 35000) & (datadf.area > 2000)]


coefficients1 = np.polyfit(clipped.angle.values, clipped.area.values, 2)

polynomial1 = np.poly1d(coefficients1)
polynomial = np.poly1d([  -17770.37753437,  32893.18466905,   4289.15777695])

xs = np.linspace(0, datadf.angle.max(), 1000)
ys = polynomial1(xs)

datadf['polydif'] = (polynomial1(datadf['angle']) - datadf['area']) 

print polynomial1
print coefficients1
plt.figure()
plt.subplot(311)
plt.hist2d(datadf.angle.values, datadf.area.values, bins=40, norm=LogNorm())
plt.colorbar()
plt.plot(xs, ys, 'yellow', linewidth=3)
plt.plot(xs, (ys/1.2)-3000.0, 'blue', linewidth=3)
plt.plot(xs, (ys*1.3)+4000.0, 'blue', linewidth=3)
ys = polynomial(xs)
plt.plot(xs, ys, 'white', linewidth=3)
plt.plot(xs, (ys/1.2)-3000.0, 'red', linewidth=3)
plt.plot(xs, (ys*1.3)+4000.0, 'red', linewidth=3)

plt.xlabel('Angle (rad)')
plt.ylabel('Wing area (pixels)')

plt.subplot(323)
plt.hist(datadf.area.values, bins=np.arange(0,30000,300))
plt.xlabel('area')
plt.ylabel('Frequency')
plt.subplot(324)
plt.hist(datadf.polydif.values, bins=np.arange(-3000,3000,50))
plt.xlabel('Difference, wing area vs polynomial1(angle)')
plt.ylabel('Frequency')
plt.text(0.95, 1000, str(flies) + ' flies\n' + str(len(datadf)) + ' wings',
         horizontalalignment='right', verticalalignment='bottom', color='black') 

plt.subplot(313)
plt.hist2d(datadf.angle.values, datadf.polydif.values, bins=100, norm=LogNorm())
plt.colorbar()
plt.xlabel('angle')
plt.ylabel('polydif')

plt.show()
