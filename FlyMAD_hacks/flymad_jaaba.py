import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pytz import common_timezones
import rosbag

#matlab data should be imported as csv with:
# column 1: seconds since epoch (16 digit precision)
# column 2: left wing angle
# column 3: right wing angle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--jaabadir', type=str, required=True,
                        help='directory of JAABA csv files')  
parser.add_argument('--outputdir', type=str, required=True,
                        help='directory to store analysis')
parser.add_argument('--bagdir', type=str, required=True,
                        help='directory of bag files')
parser.add_argument('--binsize', type=str, required=True,
                        help='integer and unit, such as "5s" or "4Min"')

args = parser.parse_args()

JAABA = args.jaabadir
BAGS = args.bagdir
OUTPUT = args.outputdir

binsize = (args.binsize)


filename = '/tier2/dickson/bathd/FlyMAD/JAABA_tracking/140927/wing_angles_nano.csv'
#binsize = '5s'  # ex: '1s' or '4Min' etc
BAG_FILE = '/groups/dickson/home/bathd/Dropbox/140927_flymad_rosbag_copy/rosbagOut_2014-09-27-14-53-54.bag'


def convert_timestamps(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = pd.to_datetime(df.pop('Timestamp'), utc=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df

def get_max(df):
    max_angle = df[['Left','Right']].abs().max(axis=1)  
    return max_angle

def bin_data(df, bin_size):
    binned = df.resample(bin_size, how='mean')  
    return binned

def parse_fmftime(namestring):
    fn = namestring.split('/')[-2]
    exp_id, CAMN, DATE, TIME = fn.split('_')
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime

def parse_bagtime(namestring):
    numstr = namestring.split('/')[-1].split('_')[-1].split('.bag')[0].replace('-','')
    bagtime = pd.to_datetime(numstr)
    return bagtime
    
def match_fmf_and_bag(fmftime):
    fmftime64 = np.to_datetime64(fmftime)
    bagtime = bagframe['Timestamp'].asof(fmftime)
    if fmftime64 - bagtime > np.timedelta64(30000000000, 'ns'):
        print "ERROR: fmf is more than 30 seconds younger than bagfile: ", fmf_dir
        continue
    else:
        bagfile = bagframe['Filepath'].asof(fmftime)
    return bagfile
    
    
    
    
def sync_jaaba_with_ros(FMF_DIR):
    JAABA_CSV = FMF_DIR + 'registered.csv'
    
    FLY_ID, FMF_TIME = parse_fmftime(FMF_DIR)
    
    BAG_FILE = match_fmf_and_bag(FMF_TIME)
    
    wing_data = pd.read_csv(JAABA_CSV, sep=',', names=['Timestamp','Width','Length','Theta','Left','Right'])
    wing_data = convert_timestamps(wing_data)
    wing_data['maxWingAngle'] = get_max(wing_data)
    #downsampled_wing_data = bin_data(wing_data, binsize)

    # extract laser info from bagfile:

    bagfile = rosbag.Bag(BAG_FILE)
    lasertimes = []
    for topic, msg, t in bagfile.read_messages('/experiment/laser'):
        lasertimes.append((t.secs +t.nsecs*1e-9,msg.data))
    lasertimes = np.array( lasertimes, dtype=[('lasertime',np.float64),('laserstate',np.float32)])
    laser_data = DataFrame(lasertimes, columns=['Timestamp', 'Laser_State'])
    laser_data = convert_timestamps(laser_data)

    wing_data['Laser_state'] = laser_data['Laser_State'].asof(wing_data.index)  #YAY! 
    wing_data['Laser_state'] = wing_data['Laser_state'].fillna(value=0)
    wing_data['Timestamp'] = wing_data.index  #silly pandas bug for subtracting from datetimeindex...
    wing_data['synced_time'] = wing_data['Timestamp'] - wing_data['Timestamp'][wing_data['Laser_state'].idxmax()]
    if not os.path.exists(JAABA + 'JAR'):
        os.makedirs(JAABA + 'JAR')
    wing_data.to_pickle(JAABA + 'JAR/' + FLY_ID + '.pickle')
    
def gather_wingdata(filelist):
    wingAng = DataFrame()
    for x in filelist:
        fx = pd.read_pickle(x)
        rel = fx[['synced_time','Laser_state', 'maxWingAngle']]
        wingAng = pd.concat(wingAng, rel)
        


baglist = []
for bag in glob.glob(BAGS + '/*.bag'):
    bagtimestamp = parse_bagtime(bag)
    baglist.append((bag, bagtimestamp))
bagframe = DataFrame(baglist, columns=['Filepath', 'Timestamp'])
bagframe.index = pd.to_datetime(bagframe['Timestamp'])
bagframe = bagframe.sort()

if not os.path.isfile(JAABA + 'JAR/*.pickle'):
    sync_jaaba_with_ros(JAABA)

if not os.path.isfile(JAABA + 'JAR/wingdata.pickle'):
    gather_wingdata(glob.glob(JAABA + 'JAR/*.pickle'))


    

