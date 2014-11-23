import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--inputfile', type=str, required=True,
                        help='must be a rawdata_blah_.pickle file') 
parser.add_argument('--inputfile2', type=str, required=False,
                        help='second file for comparative plots, range is defined by the first one')                     
parser.add_argument('--measurement', type=str, required=True,
                        help='indicate the column header you want to plot')   

parser.add_argument('--bins', type=int, required=True,
                        help='how many bins? must be an integer')   
                        

args = parser.parse_args()
inputfile = args.inputfile

print inputfile
measurement = args.measurement

num_bins=args.bins
ind = np.arange(num_bins)

print measurement, '\t', num_bins, '\t', ind

raw = pd.read_pickle(inputfile)


print measurement
    
print raw[measurement]
raw.sort(columns=measurement, ascending='False', axis=0, inplace='True')
n, bin_edges = np.histogram(raw[measurement], bins=np.linspace(min(raw[measurement]),max(raw[measurement]), num=num_bins+1))
n = n.astype('float')

print n

def make_histodata(inputdf):
    df = pd.read_pickle(inputdf)
    df.sort(columns=measurement, ascending='False', axis=0, inplace='True')
    n, bin_edges = np.histogram(df[measurement], bins=np.linspace(min(raw[measurement]),max(raw[measurement]), num=num_bins+1))
    n = n.astype('float')
    return n, bin_edges
    
labelset = np.arange(min(raw[measurement]),max(raw[measurement]),(max(raw[measurement])-min(raw[measurement]))/num_bins)
print labelset    
fig =  plt.figure()
ax2 = fig. add_subplot(1,1,1)
histo = ax2.bar(np.arange(num_bins), (n/sum(n)), 0.25, facecolor='#800000', edgecolor='gray')
if args.inputfile2:
    inputfile2 = args.inputfile2
    n, bin_edges = make_histodata(inputfile2)
    histo2 = ax2.bar(np.arange(num_bins)+0.5, (n/sum(n)), 0.25, facecolor='b', edgecolor='gray')
    

#histo3 = ax2.bar(np.arange(num_bins)+0.5, (adt2/sum(adt2)), 0.25, facecolor='green', edgecolor='gray')
ax2.set_ylabel('Frequency / total', fontsize = 16)
ax2.set_xticks(ind+0.25)
ax2.set_xticklabels(labelset, fontsize = 12, rotation=60)
ax2.set_xlabel(measurement, fontsize = 16)

ax2.legend( (histo[0], histo2[0]), (inputfile, inputfile2))

plt.show()

fig.savefig('/'.join(inputfile.split('/')[:-1]) + '/' + measurement +'.svg', bbox_inches='tight')
