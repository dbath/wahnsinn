
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse


def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id
 



def doit(_DATADIR, _HANDLE):

    fullData = pd.DataFrame({'minutes':[],'fly_x':[],'fly_y':[],'FlyID':[],'GROUP':[]})

    for directory in glob.glob(_DATADIR + '*' + _HANDLE + '*' + '*zoom*'):
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        try:
            df = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
            df['minutes'] = df.synced_time.astype('timedelta64[m]')
            tempdf = pd.DataFrame({'minutes':df.minutes, 'fly_x':df.fly_x, 'fly_y':df.fly_y, 'FlyID':FLY_ID, 'group':GROUP})
            fullData = pd.concat([fullData, tempdf], axis=0)
        except:
            pass

    fullData.to_pickle(_DATADIR + '2D_positions_' + _HANDLE + '.pickle')

    plt.figure(figsize=(24,2))

    for x in np.arange(-1,11):
        plt.subplot2grid((1,49),(0,4*(x+1)), colspan=4)
        slyce = fullData[fullData.minutes == x]
        print x, len(slyce), len(slyce.fly_x.notnull())
        plt.hist2d(slyce.fly_x.values, slyce.fly_y.values, bins=50, norm=LogNorm())
        plt.xlim(0,659)
        plt.ylim(0,494)
        plt.axis('off')
        if x == 10:
            axes = plt.subplot2grid((1,49),(0,48), colspan=1)
            plt.colorbar(cax=axes)

    plt.savefig(_DATADIR + 'dtarget_v_Length_' + _HANDLE + '.svg', bbox_inches='tight')
    return

def row_of_heatmaps(_DATADIR, _HANDLE, _treatments, _fig, row):
    for x in np.arange(0,len(_treatments)):
        datadf = pd.DataFrame({'left':[], 'right':[]})
        for f in glob.glob(_DATADIR + _HANDLE + '-UC1538-' + _treatments[x] + '*/frame_by_frame_synced.pickle'):
            df = pd.read_pickle(f)
            df['target_distance_TTM'] = df['target_distance_TTM'].fillna(method='pad',limit=2).fillna(df['dtarget'])
            df = df[(df.Length.notnull()) & (df.target_distance_TTM.notnull())]
            datadf = pd.concat([datadf, df], axis=0)
        foo = datadf[['Length','target_distance_TTM']]
        foo.columns = ['Angle','Distance']
        
        ax = _fig.add_subplot(3,4, 4*row+x+1)
        plt.hist2d(foo.Distance.values, foo.Angle.values, bins=50, norm=LogNorm())
        if row == 0:
            ax.set_title(_treatments[x])
        if x == 0:
            ax.set_ylabel(_HANDLE + '\nWing Angle (rad)')
        #ax.set_xlim(0,1.6)
        #ax.set_ylim(0,1)
    return



treatments = ['00','11','15','65']
genotypes = ['DB072','DB185','DB213']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    args = parser.parse_args()

    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'   
    fig = plt.figure()
    for x in np.arange(len(genotypes)):
        row_of_heatmaps(DATADIR, genotypes[x], treatments, fig, x)
        
    plt.show()
        
