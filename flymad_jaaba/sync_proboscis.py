    
    
    
    
    
    
    
    
    
    
    
    
    
    
def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id



prestim = np.timedelta64(5,'s')
poststim = np.timedelta64(10,'s')

def sync_by_stims(fbf, p, FlyID, GROUP):
    df = pd.merge(p, fbf, right_on='Frame_number', left_index=True)
    d = df['Laser2_state'].copy()
    t = d - d.shift()
    ts = t[abs(t) >0]
    ons = ts.index[ts == 1]
    offs = ts.index[ts == -1]
    synced_df = pd.DataFrame({'Frame_number':[],'Laser2_state':[],'length':[],'stim_number':[],'synced_seconds':[]})
    for x in range(len(ons)):
        slyce = df[ons[x]-prestim:ons[x]+poststim].copy()
        slyce['synced_time'] -= slyce.ix[0]['synced_time'] + prestim
        slyce.index = slyce['synced_time']
        slyce.index = pd.to_datetime(slyce.index)
        #slyce = slyce.resample('250ms', how='mean')
        times = []
        for y in slyce.index:
            times.append((y - pd.to_datetime(0)).total_seconds())
        slyce['synced_seconds'] = times
        slyce['stim_number'] = x+1
        l = slyce['length'].copy()
        l[l>0] = 1.0
        l = l*(l.shift() + l.shift(-1))
        l[l>0] = 1.0
        slyce['length'] *=l
        synced_df = pd.concat([synced_df, slyce[['Frame_number','Laser2_state','length','stim_number','synced_seconds']]])
    synced_df['GROUP'] = GROUP
    synced_df['FlyID'] = FlyID
    return synced_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameter', type=str, required=True,
                            help='parameter for which to measure bouts')
    #parser.add_argument('--threshold', type=float, required=True,
    #                        help='threshold to define bouts')
    #parser.add_argument('--greater_than', type=str, required=False, default="greater",
    #                        help='change to "less" to select bouts of low value')
    
    args = parser.parse_args()

    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/'    
    
    dataset = pd.DataFrame({'Frame_number':[],'Laser2_state':[],'length':[],'stim_number':[],'synced_seconds':[]})
    for directory in glob.glob(DATADIR  + '*zoom*'):
        print 'processing: ', directory.split('/')[-1]
        try:
            FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
            _fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
            _p = pd.read_pickle(directory + '/proboscis_data.pickle')
            tempdf = sync_by_stims(_fbf, _p, FLY_ID, GROUP)
            dataset = pd.concat([dataset, tempdf], axis=0)
        except:
            print "ERROR processing file."
            pass
        
        
        
        
