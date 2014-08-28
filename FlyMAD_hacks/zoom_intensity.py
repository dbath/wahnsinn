from PIL import Image, ImageStat
import glob
import matplotlib
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputdir', type=str, required=True,
                        help='directory of images')  
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to store output files')
args = parser.parse_args()
DROP = args.inputdir
OUTPUT = args.outputdir

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

fig.savefig(OUTPUT + 'plots.svg')#, bbox_inches='tight')


