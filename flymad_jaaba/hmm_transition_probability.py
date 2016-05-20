import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.colors import LogNorm
from pykalman import UnscentedKalmanFilter
from hmmlearn.hmm import GaussianHMM
from sklearn.externals import joblib
from collections import Counter
import argparse
import glob
import flymad_jaaba.utilities as utilities
import scipy.stats as ss

ukf = UnscentedKalmanFilter(initial_state_mean=0, n_dim_obs=1)



def parse_fmftime(namestring):
    fn = namestring.split('/')[-1]
    exp_id, CAMN, DATE, TIME = fn.split('_', 3)
    FLY_ID = exp_id + '_' + DATE + '_' + TIME
    fmftime = pd.to_datetime(DATE + TIME)
    return FLY_ID, fmftime, exp_id

def lineplot(grouped_df, _column, statistic):
    meansALL = grouped_df.groupby(level=[0,1]).mean()
    if 'Max' in statistic:
        semsALL = grouped_df.groupby(level=[0,1]).std()
    else:
        semsALL = grouped_df.groupby(level=[0,1]).sem()
    #set_of_groups = set(grouped_df['GROUP']))
    error_config = {'ecolor': '0.1'}
    
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = fig.add_subplot(1,1,1)
    
    adj = np.linspace(-0.025,0.025, len(grouped_df.index.levels[0]))
    
    for x in range(len(grouped_df.index.levels[0])):
        category = meansALL.index.levels[0][x]
        treatments = meansALL.index.levels[1]
        means = meansALL.ix[category]
        xs = np.arange(0, len(means.index),1) + adj[x]
        bar_width = 1.0/(len(means.index))
        sems = semsALL.ix[category]
    
        ax.errorbar(xs, 
                means[_column].fillna(0), 
                color= colourlist[x],
                yerr=sems[_column].fillna(0),
                fmt='--o',
                ecolor='#202020',
                label=category)
    
    #ax.set_ylim(0,1.3*(means[_column].values.max()))
    
    if meansALL[_column].values.min() >= 0:
        ax.set_ylim(0.98*((meansALL[_column] - semsALL[_column]).values.min()),1.1*((meansALL[_column] + semsALL[_column]).values.max()))
    else:
        ax.set_ylim(1.1*((meansALL[_column] - semsALL[_column]).values.min()),1.1*((meansALL[_column] + semsALL[_column]).values.max()))
    
    ax.margins(0.2)
    ax.set_ylabel('Mean probability ' + u"\u00B1" + ' SEM', fontsize=20)
    ax.set_xticks(np.arange(0,len(means.index),1))
    ax.set_xticklabels(list(means.index), rotation=90)
    ax.tick_params(axis='y', labelsize=16 )
    ax.set_xlabel('Stimulus', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        
    plt.legend(prop={'size':10})           
    
    plt.savefig(DATADIR + 'HMM_transition_probabilties_' + _column+'.svg', bbox_inches='tight')
    plt.close('all')
    return

def transition_matrix(Array, nstates):
    """pass a numpy array of dim (n,1) with integers representing state"""
    
    pmat = np.zeros((nstates,nstates))
    for (x,y), c in Counter(zip(Array, Array[1:])).iteritems():
        pmat[x,y] = c
    for a in range(len(pmat)):
        if not float(pmat[a].sum()) == 0:
            pmat[a] = pmat[a] / float(pmat[a].sum())
        else:
            pmat[a] = 0.0*pmat[a]
    return pmat

def smooth(df, colname):
    df[colname + '_smoothed'] = ukf.smooth(df[colname].values)[0]
    return df  

def get_hmm_bouts(state_series, _column=['state']):
    """pass a series of integers representing HMM states, with a timeseries index from fbf"""
    df = pd.DataFrame({'state':[],'mean_bout':[], 'bout_count':[], 'mean_ibi':[], 'ibi_count':[]})
    for S in list(set(state_series[_column])):
        d = state_series[_column].copy()
        d[d != S] = np.nan
        d[d == S] = 1
        d = d.fillna(0)
        t = d - d.shift()
        ONS = t[t >0].index
        OFFS = t[t < 0].index       
        if ONS[0] < OFFS[0]:
            if len(ONS) > len(OFFS):
                OFFS.append(d.index[-1])     
        elif ONS[0] > OFFS[0]:
            if len(OFFS) > len(ONS):
                ONS.insertFIXME(d.index[-1])
            bouts = (OFFS - ONS) / np.timedelta64(1,'s') #returns np array of seconds
            
            
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flymad_dir', type=str, required=True,
                            help='directory of flymad files')  
    parser.add_argument('--parameters', type=str, required=True,
                            help='parameters to define HMM')
    parser.add_argument('--n_components', type=int, required=False, default=6,
                            help='number of components in HMM')
    parser.add_argument('--pool_controls', type=bool, required=False, default=False,
                            help='Make true to pool control genotypes')
    
    args = parser.parse_args()
    parameters = [str(item) for item in args.parameters.split(',')]
    n_components = args.n_components
    DATADIR = args.flymad_dir 
    if (DATADIR[-1] != '/'):
        DATADIR = DATADIR + '/' 
        
    colourlist = ['#202020','#AAAAAA','#009020', '#6699FF', '#333333','#0032FF','r','c','m','y', '#000000']
    
    letters = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    tmatcolumns = ['FlyID','group'] + [i+j for i in letters[0:n_components] for j in letters[0:n_components]]
    transmat = pd.DataFrame(columns = tmatcolumns)
    statecolumns = ['FlyID','group'] + letters[0:n_components]#[i + j for i in letters[0:n_components] for j in parameters]
    state_means = pd.DataFrame(columns = statecolumns)

    exampleFBF = pd.read_pickle('/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160224_094231/frame_by_frame_synced.pickle')
    
    model_builders = [
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160223_124401/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160224_094231/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160224_190311/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160225_122636/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160303_181631/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160303_194430/frame_by_frame_synced.pickle',
                      '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB213-UC1538-15_zoom_20160301_114025/frame_by_frame_synced.pickle']
    Xmodel_builders = [
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160307_112535/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160301_174810/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160308_182658/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160223_125615/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160303_182857/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160308_172623/frame_by_frame_synced.pickle',
                    '/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/DB072-UC1538-15_zoom_20160308_124602/frame_by_frame_synced.pickle']

    list_of_datasets=[]
    lengths=[]
    for fn in model_builders:
        try: 
            fbf = pd.read_pickle(fn)
            x_ = np.column_stack(fbf[ i +'_smoothed'] for i in parameters)
            list_of_datasets.append(x_)
            lengths.append(len(x_))
        except: 
            print 'failed to complete: ', grp, fn
    values = np.concatenate(list_of_datasets)    
    #values = np.load('/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/HMM_JAR/DB213-UC1538-15_X.npy' )
    #lengths = np.load('/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/HMM_JAR/DB213-UC1538-15_lengths.npy' )
    #eX = np.column_stack(exampleFBF[ i +'_smoothed'] for i in parameters)
    
    
    """
    bics = []
    aics = []
    for n in range(1,15):
        THE_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000).fit(values, lengths)
        print "**      N = ", n
        print THE_model.means_
        L = THE_model.score(x_)
        l = len(values)
        bic = -2.0*L + float(n)*np.log(l)
        aic = 2.0*float(n) - 2.0*(L)
        print "BIC: ", L, l, bic, aic
        bics.append(bic)
        aics.append(aic)
    plt.plot(range(1,15), bics/max(bics))
    #plt.plot(range(1,15), aics/max(aics))
    plt.show()
    
    """
    
    THE_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000).fit(values, lengths)

    print THE_model.means_
    print pd.DataFrame(THE_model.transmat_)

    wing_vs_state = np.array([np.nan,np.nan,np.nan])
    for directory in glob.glob(DATADIR  + '*zoom*'):
        FLY_ID, FMF_TIME, GROUP = parse_fmftime(directory)
        try:
            fbf = pd.read_pickle(directory + '/frame_by_frame_synced.pickle')
        except:
            continue
        if 'target_distance_TTM' in fbf.columns:
            fbf['target_distance_TTM'] = fbf['target_distance_TTM'].fillna(method='pad',limit=3).fillna(fbf['dtarget']).fillna(fbf['dtarget'].max())
        if 'group' in fbf.columns:
            GROUP = fbf['group'][0]
        fbf.loc[fbf['Length'] <= 245.0,'Length'] = np.nan
        for parameter in parameters:
            if parameter == 'maxWingAngle':
                fillValue = 0.0
            elif parameter == 'target_distance_TTM':
                fillValue = 30.0
            elif parameter == 'Length':
                fillValue = 325.0
            else:
                fillValue = 0
            if (parameter + '_smoothed') not in fbf.columns:
                fbf[parameter] = fbf[parameter].fillna(method='pad', limit=5).fillna(fillValue)
                fbf = smooth(fbf, parameter)
                fbf.to_pickle(directory + '/frame_by_frame_synced.pickle')
        
        #CREATE HIDDEN MARKOV MODEL
        
        _fbf = fbf.loc[fbf['synced_time'] > np.timedelta64(0,'ns')]  #take only post-stimulus data
        X = np.column_stack(_fbf[ i +'_smoothed'] for i in parameters)
        
        state_values = pd.DataFrame(THE_model.predict(X), columns=['state'])
        #DISCARD CASES WHERE ONE OR MORE STATES OCCURS RARELY (<1%).
        DISCARD = False
        for i in list(set(state_values['state'])):
            if (len(state_values[state_values['state']==i]) / float(len(state_values)) < 0.005) & (len(state_values[state_values['state']==i]) >0):
                print i, len(state_values), len(state_values[state_values['state'] == i]), '\t', FLY_ID
                state_values.loc[state_values['state']==i, 'state'] = np.nan
                #DISCARD = True
        state_values['state'] = state_values['state'].fillna(method='pad').fillna(method='bfill')
        state_values = np.array(state_values['state']) 
        
        statesdf = pd.DataFrame(state_values, columns=['state'], index = _fbf.index)
        
        bouts = get_hmm_bouts(statesdf)
        
        transmat_temp = transition_matrix(state_values, n_components)
        transmat = transmat.append(pd.Series([FLY_ID, GROUP] + list(transmat_temp.ravel()) , 
                                 index=tmatcolumns), ignore_index=True)
        state_means = state_means.append(pd.Series([FLY_ID, GROUP] + [float(len(state_values[state_values==x]))/float(len(state_values)) for x in np.arange(n_components)],index=statecolumns), ignore_index=True)
        
        wing_vs_state = np.row_stack([wing_vs_state,np.column_stack([state_values,X])])
            
            
    for x in transmat.index:
        g =  transmat.loc[x, 'group'].split('-')
        transmat.loc[x, 'g1'] = g[0]
        try:
            transmat.loc[x, 'g2'] = g[1]
        except:
            transmat.loc[x, 'g2'] = 'nix'
        try:
            transmat.loc[x, 'g3'] = g[2]
        except:
            transmat.loc[x, 'g3'] = 'nix'

    transmat.to_pickle(DATADIR + 'HMM_transition_probabilties_'+ str(n_components) +'.pickle')
    transmat.to_csv(DATADIR + 'HMM_transition_probabilties_'+ str(n_components) +'.csv', sep=',')
    state_means.to_pickle(DATADIR + 'HMM_state_means_'+ str(n_components) +'.pickle')
    state_means.to_csv(DATADIR + 'HMM_state_means_'+ str(n_components) +'.csv', sep=',')
    print state_means.groupby('group').mean()
    
    if args.pool_controls:
        transmat.loc[transmat['g1'] =='DB072', 'g1'] = 'ctrl'
        transmat.loc[transmat['g1'] =='DB185', 'g1'] = 'ctrl'
    p_values = pd.DataFrame()
    grouped = transmat.groupby(['g1','g3','FlyID'])
    for t in tmatcolumns[2:]:
        lineplot(grouped.mean(), t, t)        
        p_values = pd.concat([p_values, utilities.stats_pairwise(transmat, t, 'g1','g3')], axis=0)
        p_values = pd.concat([p_values, utilities.stats_pairwise(transmat, t, 'g3','g1')], axis=0)

    p_values.to_pickle(DATADIR + 'HMM_state_pvalues_'+ str(n_components) +'.pickle')
    p_values.to_csv(DATADIR + 'HMM_state_pvalues_'+ str(n_components) +'.csv', sep=',')
        
    signif = p_values[p_values['p'] < 0.05]
    print signif 
    
    
    wVS = pd.DataFrame(wing_vs_state, columns=['state','distance','angle']).dropna() 
    wVS['distance'] = wVS['distance'].max() - wVS['distance']
    
    abins = np.linspace(0, 2*np.pi, 360)
    sbins = np.linspace(-1000, 50, 2)
    theta, r = np.mgrid[0:2*np.pi:360j, wVS['distance'].min():wVS['distance'].max():2j]

    
    wVS.to_pickle('/tier2/dickson/bathd/FlyMAD/data_for_processing/160223_adt2sil_DR/HMM_groups_test.pickle')
    colours = cm.rainbow(np.linspace(0, 1, n_components))   
    colours = ["#0000FF", "#FF0000"]


        
        
