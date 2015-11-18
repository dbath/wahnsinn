
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id
 



fullData = pd.DataFrame({'minutes':[],'fly_x':[],'fly_y':[],'FlyID':[],'GROUP':[]})


DATADIR = '/media/DBATH_10/151029_P1_adt2_coact/'
HANDLE = 'DB139'

for directory in glob.glob(DATADIR + '*' + HANDLE + '*' + '*zoom*'):
    FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
    try:
        df = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        df['minutes'] = df.synced_time.astype('timedelta64[m]')
        tempdf = pd.DataFrame({'minutes':df.minutes, 'fly_x':df.fly_x, 'fly_y':df.fly_y, 'FlyID':FLY_ID, 'group':GROUP})
        fullData = pd.concat([fullData, tempdf], axis=0)
    except:
        pass

fullData.to_pickle(DATADIR + HANDLE + '_position_data.pickle')

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

plt.savefig(DATADIR + 'position_plots_' + HANDLE + '.svg', bbox_inches='tight')
