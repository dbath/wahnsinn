
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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

    plt.savefig(_DATADIR + 'position_plots_' + _HANDLE + '.svg', bbox_inches='tight')
    return

def row_of_heatmaps(_DATADIR, _HANDLE, _treatments, _fig, row):
    for x in np.arange(0,len(_treatments)):
        datadf = pd.DataFrame({'left':[], 'right':[]})
        for f in glob.glob(_DATADIR + _HANDLE + '-UC1538-' + _treatments[x] + '*/tracking_info.pickle'):
            df = pd.read_pickle(f)
            df.index.names=['Frame_number']
            try:
                fbf = pd.read_pickle(f.rsplit('/',1)[0] + '/frame_by_frame_synced.pickle')
                fbf['dtarget'] = fbf['target_distance_TTM_smoothed']
                ds = fbf['dtarget']
                ds.index = fbf['Frame_number']
                df = pd.concat([df,ds], axis=1, join='outer')
            except:
                continue
            df = df[(df.a_wingAngle_left.notnull()) & (df.b_wingAngle_right.notnull())]
            datadf = pd.concat([datadf, df], axis=0)
        foo = datadf[['a_wingAngle_left','b_wingAngle_right']]
        foo.columns = ['left','right']
        foo['maxAngle'] = foo.max(axis=1)
        foo['minAngle'] = foo.min(axis=1)
        foo['laterality'] = (foo['maxAngle'] - foo['minAngle'] )* foo['maxAngle']
        foo['dtarget'] = datadf['dtarget']
        foo.to_pickle(_DATADIR + _HANDLE + '-UC1538-' + _treatments[x] + '_laterality_scores.pickle')
        
        coefficients = np.polyfit(foo.maxAngle.values, foo.laterality.values, 2)
        coeff_07200 = [ 0.99006412, -0.1036647,   0.0041638 ]

        polynomial = np.poly1d(coeff_07200)
        
        print _HANDLE, _treatments[x], '\n', polynomial, '\n', coefficients
        
        foo = foo[foo.laterality < (polynomial(foo.maxAngle) - 0.4)]
        
        ax = _fig.add_subplot(3,4, 4*row+x+1)
        
        plt.hist2d(foo.maxAngle.values, foo.laterality.values, bins=50, norm=LogNorm())
        if row == 0:
            ax.set_title(_treatments[x])
        if row == 2:
            ax.set_xlabel('Wing Angle')
        if x == 0:
            ax.set_ylabel(_HANDLE + '\nLaterality score')
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
        
