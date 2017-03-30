import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from hmmlearn.hmm import MultinomialHMM, GaussianHMM
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def process_file(fn):
    #get the data
	df = pd.read_table(fn,sep=',', header=0, skiprows=[0,1,3])
	df = df.sort('Time').reset_index() 
	full = df[['Time','TrackName']] 

	#Discard data after copulation	
	if len(full[full.TrackName == 'copulation']) > 0:
		full = full[0:full[full.TrackName == 'copulation'].index[0]]
	
	#Replace scoring names with behaviour strings
	for x in range(len(scores)):
		full['TrackName'] = full['TrackName'].str.replace(scores[x], behaviours[x]) 	


	#Extract just the behaviours of interest, ignoring others
	full = full[full['TrackName'].isin(behaviours)]

	#Annotate previous and next behaviours (relevant only for latencies)
	full['next'] = full.shift(-1)['TrackName']
	full['previous'] = full.shift(1)['TrackName']
	full.columns = ['Time','current','next', 'previous']
	full = full.reset_index()
	return df, full



def strings_to_ints(col):
    '''convert dataframe column with strings to integers'''
    names = list(set(col))
    names.sort()
    numlist = []
    for x in col:
        numlist.append(names.index(x))
    return numlist
    


def get_transmat(model, values, lengths, _type):
    "returns the transition matrix with sorted axes as dataframe or numpy array"
    #fit a markov model to the data
    model.fit(np.atleast_2d(values).T, lengths)
    #define the index (gets randomized by model.fit())
    tindex =  [str(model.means_[x][0]) for x in range(len(model.means_))]
    #get the transition matrix
    tdf = model.transmat_
    #make the transition matrix legible
    tdf = pd.DataFrame(tdf, columns=tindex, index=tindex)
    tdf = tdf.sort_index(axis=0).sort_index(axis=1)
    #return either numpy array or pandas dataframe
    if _type == 'df':
        return tdf
    elif _type == 'array':
        return np.array(tdf)		
	
	
def permute_CI_values(model, values, lengths, reps):
    """
    Performs Monte Carlo permutation analysis given a Markov chain model.
    Returns: t (the transition matrix),
             c (a matrix of probabilities that each event occurs less than
                predicted by random chance)
             d (a matrix of probabilities that each event occurs more than
                predicted by random chance)
    """
    T = get_transmat(model, values, lengths, 'array')
    C = np.zeros(T.shape) # accumulates where B>=T
    D = np.zeros(T.shape) # accumulates where C<=T
    for x in range(reps):
        rand = np.random.permutation(values) #randomize
        B = get_transmat(model, rand, lengths, 'array') #generate transition matrix from random sequence
        C = (B>=T) + C  
        D = (B<=T) + D
    return T, C/(reps*1.0), D/(reps*1.0)


#DEFINE DATASET AND WHICH BEHAVIOURS TO INCLUDE IN THE ANALYSIS		
DATAFILES = '/home/dbath/Documents/170322_claire_vcode_data/*-evts.txt'
scores = ['wing extension','medPE','attempted copulation','bothlegs']
behaviours = ['a_wingExt','b_probExt','c_abBend','d_legLift']


#DO THE ANALYSIS FOR EACH EXPERIMENT FILE
list_of_datasets=[]
lengths=[]
for _fn in glob.glob(DATAFILES):
    dataset, fulldf = process_file(_fn)
    fulldf['intlist'] = strings_to_ints(fulldf['current'].copy())
    list_of_datasets.append(fulldf['intlist'])
    lengths.append(len(fulldf))
    
observed_states = np.concatenate(list_of_datasets) 

#USE THE SEQUENCES TO CREATE A MARKOV MODEL
model = GaussianHMM(n_components=len(behaviours), 
                    n_iter=1000).fit(np.atleast_2d(observed_states).T, lengths)

#COMPARE THE ACTUAL SEQUENCE TO RANDOMIZED SEQUENCE WITH A PERMUTATION TEST
t, c, d = permute_CI_values(model, observed_states, lengths, 10000)

#DISPLAY THE RESULTS
print behaviours
print "Transition Matrix: \n", t
print "CI(transition less than null hypothesis: \n", c
print "CI(transition more than null hypothesis: \n", d

#SAVE THE RESULTS
t.to_csv(DATAFILES.rsplit('/',1)[-1] + '/transition_matrix.csv')
c.to_csv(DATAFILES.rsplit('/',1)[-1] + '/CI_t_less_than_random.csv')
d.to_csv(DATAFILES.rsplit('/',1)[-1] + '/CI_t_more_than_random.csv')


"""
RESULTS:

['a_wingExt', 'b_probExt', 'c_abBend', 'd_legLift']

Transition Matrix:  

[[ 0.6872428   0.26337449  0.04115226  0.00823045]
 [ 0.26573427  0.30769231  0.34965035  0.07692308]
 [ 0.20833333  0.25        0.08333333  0.45833333]
 [ 0.2         0.38181818  0.14545455  0.27272727]]
 
CI(transition less than null hypothesis:

[[  0.     0.7837   1.       1.    ]
 [  1.     0.1835   0.       0.9616]
 [  1.     0.7156   0.9233   0.    ]
 [  1.     0.0367   0.4737   1.    ]]
 
CI(transition more than null hypothesis:

[[ 1.      0.2163  0.      0.    ]
 [ 0.      0.8167  1.      0.0384]
 [ 0.      0.285   0.0768  1.    ]
 [ 0.      0.9633  0.5263  0.9999]]

"""

