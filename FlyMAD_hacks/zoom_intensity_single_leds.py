from PIL import Image, ImageStat
import glob
import matplotlib
import matplotlib.pyplot as plt

DROP = '/groups/dickson/home/bathd/Dropbox/140827_no_arena_zoom/'
images = sorted(glob.glob(DROP + '*.png'))


medians = []
totals = []

for x in images:
    im = Image.open(x)
    med = ImageStat.Stat(im).median
    total = ImageStat.Stat(im).sum
    medians.append(med)
    totals.append(total)

fig = plt.figure()  
ax = fig.add_subplot(2,1,1)
plt.plot(medians, color='g', linewidth=2)

plt.xlabel('Position', fontsize=12)
plt.ylabel('Median pixel intensity (AU)', fontsize=12)

ax2 = fig.add_subplot(2,1,2)
plt.plot(totals, color='g', linewidth=2)

plt.xlabel('Position', fontsize=12)
plt.ylabel('Sum pixel intensity (AU)', fontsize=12)

plt.show()

fig.savefig(DROP + 'plots.svg')#, bbox_inches='tight')


