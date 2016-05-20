import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from pykalman import UnscentedKalmanFilter
from hmmlearn.hmm import GaussianHMM

ukf = UnscentedKalmanFilter(initial_state_mean=0, n_dim_obs=1)


def smooth(df, colname):
    df[colname + '_smoothed'] = ukf.smooth(df[colname].values)[0]
    return df  

def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id
    

treatments = ['00','11','15','65']
genotypes = ['DB072','DB185','DB213']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameters', type=list, required=True,
                            help='parameters to define HMM')
    parser.add_argument('--n_components', type=int, required=False, default=6,
                            help='number of components in HMM')
    
    args = parser.parse_args()
    parameters = args.parameters
    n_components = args.n_components
    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/' 
    dataset = pd.DataFrame({'FlyID': [], 'group':[], 'tmatrix':[], 'kOFF':[]})
    ID = []
    groups = []
    tmatrices = []
    component_means = []
    filelist = []      
    for directory in glob.glob(DATADIR  + '*zoom*'):
        try:
            fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        except:
            continue
        
        #  DATA SCRUBBING
        
        if 'target_distance_TTM' in fbf.columns:
            fbf['target_distance_TTM'] = fbf['target_distance_TTM'].fillna(method='pad',limit=3).fillna(fbf['dtarget']).fillna(fbf['dtarget'].max())
        fbf.loc[fbf['Length'] <= 245.0,'Length'] = np.nan

        for parameter in parameters:
            if (parameter + '_smoothed') not in fbf.columns:
                fbf[parameter] = fbf[parameter].fillna(method='pad', limit=5).fillna(fillValue)
                fbf = smooth(fbf, parameter)
                fbf.to_pickle(directory + '/frame_by_frame_synced.pickle')

        # EXPERIMENT SORTING: 
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        if 'group' in fbf.columns:
            GROUP = fbf['group'][0]
        ID.append(FLY_ID)
        groups.append(GROUP)        
        filelist.append(directory+ '/frame_by_frame_synced.pickle')
    filesorter = pd.DataFrame({'group':groups, 'filepath':filelist})
    
    # MAKE ONE MODEL PER EXPERIMENTAL GROUP:
    
    groups = []
    likelihoods = []
    for grp, files in filesorter.groupby('group'):    
        list_of_datasets = []   
        lengths = []
        for fn in files.index:
            try: 
                fbf = pd.read_pickle(files.ix[fn]['filepath'])
                x_ = np.column_stack(fbf[ i +'_smoothed'] for i in parameters)
                list_of_datasets.append(x_)
                lengths.append(len(x_))
            except: 
                print 'failed to complete: ', grp, fn
        X = np.concatenate(list_of_datasets)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_X.npy', X)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_lengths.npy', lengths)
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X, lengths)
        likelihoods.append(model.score(X))
        joblib.dump(model, DATADIR + 'HMM_JAR/' + grp +'_model.pkl')
        groups.append(grp)

    #  MAKE ONE MODEL PER TREATMENT:

    for grp in treatments:
        
        list_of_datasets = [] 
        for data in glob.glob( DATADIR + 'HMM_JAR/*'+ grp +'_X.npy'):
            list_of_datasets.append(np.load(data))
        X = np.concatenate(list_of_datasets)
        lengths = []
        for l in glob.glob( DATADIR + 'HMM_JAR/*'+ grp +'_lengths.npy'):
            lengths.append(np.load(l))
        lengths = np.concatenate(lengths)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_XTMT.npy', X)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_lengthsTMT.npy', lengths)
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X, lengths)
        likelihoods.append(model.score(X))
        joblib.dump(model, DATADIR + 'HMM_JAR/' + grp +'_modelTMT.pkl')
        groups.append(grp)            
        
    #  MAKE ONE MODEL PER GENOTYPE:

    for grp in genotypes:
        print 'processing: ', grp
        list_of_datasets = [] 
        for data in glob.glob( DATADIR + 'HMM_JAR/'+ grp +'*_X.npy'):
            list_of_datasets.append(np.load(data))
        X = np.concatenate(list_of_datasets)
        lengths = []
        for l in glob.glob( DATADIR + 'HMM_JAR/'+ grp +'*_lengths.npy'):
            lengths.append(np.load(l))
        lengths = np.concatenate(lengths)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_XGN.npy', X)
        np.save(DATADIR + 'HMM_JAR/' + grp +'_lengthsGN.npy', lengths)
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X, lengths)
        likelihoods.append(model.score(X))
        joblib.dump(model, DATADIR + 'HMM_JAR/' + grp +'_modelGN.pkl')
        groups.append(grp)    
    full_models = pd.DataFrame({'group':groups, 'likelihood':likelihoods})   
    
    # TEST EACH INDIVIDUAL MODEL AGAINST TREATMENT-WIDE MODEL:
    
    comparisons = []
    test_l = []
    for grp in treatments:
        print "processing: ", grp
        test_model = joblib.load(DATADIR + 'HMM_JAR/' + grp + '_modelTMT.pkl')
        for data in glob.glob(DATADIR + 'HMM_JAR/*' + grp + '_X.npy'):
            groupname = data.split('/')[-1].split('_X.npy')[0] + '_vs_' + grp
            comparisons.append(groupname)
            test_l.append(test_model.score(np.load(data)))

    # TEST EACH INDIVIDUAL MODEL AGAINST GENOTYPE-WIDE MODEL:
    for grp in genotypes:
        print "processing: ", grp
        test_model = joblib.load(DATADIR + 'HMM_JAR/' + grp + '_modelGN.pkl')
        for data in glob.glob(DATADIR + 'HMM_JAR/' + grp + '*_X.npy'):
            groupname = data.split('/')[-1].split('_X.npy')[0] + '_vs_' + grp
            comparisons.append(groupname)
            test_l.append(test_model.score(np.load(data)))
            
    test_models = pd.DataFrame({'group':comparisons, 'likelihood':test_l})         
            
            
            
    
    
    
      
        
        
        
        
        for fn in files.index:
            try:
                fbf = pd.read_pickle(files.ix[fn]['filepath'])
                x_ = np.column_stack(fbf[ i +'_smoothed'] for i in parameters)
                list_of_datasets.append(x_)
                lengths.append(len(x_))
            except: pass
        X = np.concatenate(list_of_datasets)
        np.save(DATADIR + '/HHM_JAR/' + grp +'_X.npy', X)
        np.save(DATADIR + '/HHM_JAR/' + grp +'_lengths.npy', lengths)
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X, lengths)
        likelihoods.append(model.score(X))
        joblib.dump(model, DATADIR + '/HHM_JAR/' + grp +'_model.pkl')
        groups.append(grp)
    single_models = pd.DataFrame({'group':groups, 'likelihood':likelihoods, 'model_fn':
        
   
    likelihoods = []
    l_tmt = []
    for group in treatments:    
        list_of_datasets = []   
        lengths = []     
        for directory in glob.glob(DATADIR + '*' + group + '*'):
            try:
                fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
                x_ = np.column_stack(fbf[ i +'_smoothed'] for i in parameters)
                list_of_datasets.append(x_)
                lengths.append(len(x_))
            except: pass
        X = np.concatenate(list_of_datasets)
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X, lengths)
        l_tmt.append(model.score(X))

        """
        
        #  GAUSSIAN HIDDEN MARKOV MODEL:
        
        X = np.column_stack(fbf[ i +'_smoothed'] for i in parameters)    
        
        model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(X)
        print grp
        print model.transmat_
        hidden_states = model.predict(X)
        tmatrices.append(model.transmat_)
        component_means.append(model.means_)
        

