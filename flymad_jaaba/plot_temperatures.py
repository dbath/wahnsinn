import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df


if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', type=str, required=True,
                            help='directory of dat files')  
    args = parser.parse_args()
    _DIR = args.folder
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    counter = 0
    colourlist = ['b','g','y','orange','r']
    for x in sorted(glob.glob(_DIR + '*.dat')):
        df = pd.read_table(x, skiprows=[0], names = ['Timestamp', 'v', 'temp', 'l1', 'l2'], sep='\t')
        df = convert_timestamps(df)
        df['Timestamp'] = df.index
        df['synced_time'] = df['Timestamp'] - df.Timestamp[df.l2 > 0].index[0]
        df.temp = df.temp - df.temp[df.synced_time < 0].mean() + 24.5
        print (str(df.l1[df.l1>0].mean()) + 'A'), '\t', df['Timestamp'][df.l1 > 0].index[0]
        p = plt.plot(df['synced_time'], df['temp'],
                    label = (str(df.l1[df.l1>0].mean()) + 'A'), 
                    color = colourlist[counter]
        )
        counter +=1
        
        
        
    ax.set_ylabel('Temperature (C)')
    ax.set_xlabel('Time (min)')
        
    plt.show()
    plt.savefig(_DIR + 'figure.svg')
