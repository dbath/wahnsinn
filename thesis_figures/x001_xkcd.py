import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


raw = pd.read_csv('/groups/dickson/home/bathd/Dropbox/THESIS/X001_P1VT_screen/raw_data.csv', sep=',')
raw.sort(columns='score', ascending='False', axis=0, inplace='True')

ctrl_subset = raw[raw['vt_line'] == 'No-driver']

num_bins=10
n, bin_edges = np.histogram(raw['score'], bins=np.linspace(0,1, num=num_bins+1))
n = n.astype('float')
ctrl, blah2 = np.histogram(ctrl_subset['score'], bins=np.linspace(0,1, num=num_bins+1))
ctrl = ctrl.astype('float')
ind = np.arange(num_bins)

labelset = []
for x in ind:
    bin_name = str(bin_edges[x]) + " - "+ str(bin_edges[x+1])
    labelset.append(str(bin_name))
      

with plt.xkcd():


    fig =  plt.figure()

    ax = fig.add_subplot(2,1,1)
    rects = ax.bar((len(raw)-(np.arange(len(raw)))), raw['score'], 1, edgecolor='gray', facecolor='#800000') #HACK
    ax.set_ylabel('Proportion extending wing', fontsize = 16)
    ax.set_title('P1 co-activation screen summary')
    ax.set_xticks((len(raw)-(np.arange(len(raw))))) #HACK
    ax.set_xticklabels(raw['vt_line'], rotation='vertical', fontsize=8 )
    ax.set_ylim(0,1.2)
    plt.subplots_adjust(bottom = 0.1, top=0.95, left=0.05, right=0.95)
        
    
    ax2 = fig. add_subplot(2,1,2)
    histo = ax2.bar(np.arange(num_bins), (n/sum(n)), 0.33, facecolor='#800000', edgecolor='gray')
    histo2 = ax2.bar(np.arange(num_bins)+0.33, (ctrl/sum(ctrl)), 0.33, facecolor='#0099CC', edgecolor='gray')
    ax2.set_ylabel('Frequency / total', fontsize = 16)
    ax2.set_xticks(ind+0.33)
    ax2.set_xticklabels(labelset, fontsize = 12)
    ax2.set_xlabel('Proportion of flies extending wings', fontsize = 16)

    ax2.legend( (histo[0], histo2[0]), ('Full dataset', 'Controls only'))

plt.show()

fig.savefig("/groups/dickson/home/bathd/Dropbox/THESIS/X001_P1VT_screen/figure_X001.svg", bbox_inches='tight')
