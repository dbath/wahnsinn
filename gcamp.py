
import pandas as pd
import numpy as np
import argparse
from matplotlib import cm 
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.collections as collections
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def twin_axis_plot(xvar, var1, var2, xlabel, label1, label2):
    ax1 = plt.gca()# = plt.subplots()
    ax1.plot(xvar, var1, 'b-')
    ax1.set_xlabel(xlabel)
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel(label1, color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.plot(xvar, var2, 'r.')
    ax2.set_ylabel(label2, color='r')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        tick.label.set_rotation('vertical')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    return 



def plot_data(means, sems, ns, measurement, ax):
    means = means[means[measurement].notnull()]
    #means = 1.0 - means
    #ns = ns[ns[measurement].notnull()]
    #sems = sems[sems[measurement].notnull()]
    #fig = plt.figure()
    group_number = 0
    #ax = fig.add_subplot(1,1,1)
    y_range = []
    x_range = []
    laser_x = []
    for x in means.index.levels[0]:
        max_n = ns.ix[x]['ExpID'].max()
        x_values = []
        y_values = []
        psems = []
        nsems = []
        for w in means.ix[x].index:
            laser_x.append((w-pd.to_datetime(0)).total_seconds())
            if ns.ix[x]['ExpID'][w] > ((max_n)-3): #(max_n/3):
                x_range.append((w-pd.to_datetime(0)).total_seconds())
                #print ns.ix[x]['FlyID'][w]
                x_values.append((w-pd.to_datetime(0)).total_seconds())
                y_values.append(means.ix[x,w][measurement])
                psems.append(sems.ix[x,w][measurement])
                nsems.append(-1.0*sems.ix[x,w][measurement])
        #x_values = list((means.ix[x].index - pd.to_datetime(0)).total_seconds())
        #y_values = list(means.ix[x][measurement])
        #psems = list(sems.ix[x][measurement])
        #nsems = list(-1*(sems.ix[x][measurement]))
        y_range.append(np.amin(y_values))
        y_range.append(np.amax(y_values))
        top_errbar = tuple(map(sum, zip(psems, y_values)))
        bottom_errbar = tuple(map(sum, zip(nsems, y_values)))
        p = plt.plot(x_values, y_values, linewidth=3, zorder=100,
                        linestyle = '-',
                        color=colourlist[group_number],
                        label=(str(x) + ', n= ' + str(max_n))) 
        q = plt.fill_between(x_values, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.15, 
                            linewidth=0,
                            zorder=90,
                            color=colourlist[group_number],
                            )
        group_number += 1
    if 'maxWingAngle' in measurement:
        ax.set_ylabel('Mean maximum wing angle (rad)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
    elif 'dtarget' in measurement:
        ax.set_ylabel('Mean min. distance to target (mm)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
        
    else:
        ax.set_ylabel('Mean ' + measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Time (s)', fontsize=16)      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.legend()
    return




############################################multi-colour tools   ########################################################################

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, zorder=100):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    ax = plt.gca()
    ax.set_xlim(0.85*x.min(), 1.15*x.max())
    ax.set_ylim(1.15*y.min(), 1.15*y.max())
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 



############################################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', type=str, required=True,
                        help='path and filename of csv containing experiment info')
    parser.add_argument('--savedir', type=str, required=False, default='undefined',
                help='path to save directory, default is same directory as experiment file')

    args = parser.parse_args()
    
    if args.savedir == 'undefined':
        SAVEDIR = args.experiment.rsplit('/',1)[0]
    else:
        SAVEDIR = args.SAVEDIR

    if not SAVEDIR[-1] == '/':
        SAVEDIR = SAVEDIR + '/'




    read_column = 'Intensity Region 1'
    background_column = 'Intensity Region 2'
    BASELINE_DURATION = 10 #seconds
    SAMPLING_RATE = '500ms'

    jdict = pd.read_csv(args.experiment)
    jdict = jdict[jdict['GENOTYPE'].notnull()]
    jdict['calcfn'] = ['/groups/dickson/dicksonlab' + z.split('Z:')[1] for z in jdict['calcfn'].values]
    jdict = jdict[jdict['DATE'] >= 160919]
    jdict['notes'] = jdict['notes'].fillna('')
    #jdict = jdict[jdict['notes'].str.contains('Acetylcholine')]
    jdict = jdict[jdict['include'] == 1].reset_index()
    datadf = pd.DataFrame()

    for expNum in range(len(jdict)):
        #try:
        calcdata = pd.read_table(jdict['calcfn'][expNum])
        print jdict['calcfn'][expNum]
        filename = '-'.join((jdict['calcfn'][expNum]).split('/')[-3:]).rsplit('.',1)[0] + '.png'
        #filename = '-'.join((jdict['calcfn'][expNum]).split(SAVEDIR.split('/')[-2])[1].split('/')).split('.txt')[0] + '.png'
        genotype = jdict['GENOTYPE'][expNum]
        calccolumn = calcdata.columns[jdict['calcColNum'][expNum]-1]
        bgcol = calcdata.columns[jdict['bgColNum'][expNum]-1]
        #lightcolumn = jdict['lightColName'][expNum]
        
        calcdata = calcdata[:-1]
        calcdata.index = pd.to_datetime(calcdata['Time [s]'], unit='s')
        calcdata['df'] = calcdata[calccolumn] - calcdata[bgcol]
        baseline = calcdata[calcdata.index <= pd.to_datetime(BASELINE_DURATION*1E9)]['df'].mean()
        calcdata['dff'] = (calcdata['df'] - baseline) / baseline

        
        tempdf = calcdata[['Time [s]','dff']].resample(SAMPLING_RATE)
        tempdf.columns = ['Time','dff']
        tempdf['ExpID'] = filename.split('.png')[0]
        tempdf['Genotype'] = genotype
        datadf = pd.concat([datadf,tempdf], axis=0)
        
        
        fig = plt.figure()
        axA = fig.add_subplot(111)
        twin_axis_plot(tempdf.index, tempdf['dff'], tempdf['dff'], 'Time (hh:mm:ss)', 'dF/F0', 'dF/F0')
        
        plt.savefig(SAVEDIR+filename)
        plt.close('all')
        '''
        except:
            print 'ERROR PROCESSING:', jdict['calcfn'][expNum]
            continue
        
        '''
    g = datadf.groupby(['Genotype',datadf.index])

    g.mean().unstack(level=0)['dff'].plot()
    plt.xlim(pd.to_datetime(0),pd.to_datetime(600*1e9))

    plt.show()

    means = g.mean()
    sems = g.sem()
    ns = g.count() 
    colourlist = ['#FF0000','#009020','#0000CC']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot_data(means,sems, ns, 'dff',ax1)

    t = datadf.groupby([datadf.index]).mean().reset_index()
    ax1.set_xlabel('Time')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('dF/F0', color='k')
    ax1.set_yscale('symlog')
    plt.axhline(y=0, linestyle='--', color='k')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    if 'Light' in t.columns:
        ax2 = ax1.twinx()
        print t.columns
        ax2.plot(t['Time'], t['Light'], 'k.')
        ax2.set_ylabel('Laser', color='k')
        #ax2.set_ylim(0,32)
    plt.show() 




