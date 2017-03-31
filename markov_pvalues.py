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
    
def delete_repeats(v, l):
    arrays = np.split(v, np.cumsum(l)[:-1])
    new_lengths = []
    new_v = np.array([])
    for x in range(len(arrays)):
        vals = np.append(arrays[x][:-1][np.diff(arrays[x]) != 0], arrays[x][-1])
        new_lengths.append(len(vals))
        new_v = np.append(new_v, vals)
    return new_v, new_lengths
        
def get_transmat(model, values, lengths, _type):
    "returns the transition matrix with sorted axes as dataframe or numpy array"
    if DELETE_REPEATS:
        values, lengths = delete_repeats(values, lengths)
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
    D = np.zeros(T.shape) # accumulates where B<=T
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

REVERSED = True
DELETE_REPEATS = True

#DO THE ANALYSIS FOR EACH EXPERIMENT FILE
list_of_datasets=[]
lengths=[]
for _fn in glob.glob(DATAFILES):
    dataset, fulldf = process_file(_fn)
    fulldf['intlist'] = strings_to_ints(fulldf['current'].copy())
    if REVERSED:
        list_of_datasets.append([i for i in reversed(fulldf['intlist'])])
    else:
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
idt = ['W','P','A','L']
pd.DataFrame(t, columns = idt, index=idt).to_csv(DATAFILES.rsplit('/',1)[0] + '/transition_matrix.csv')
pd.DataFrame(c, columns = idt, index=idt).to_csv(DATAFILES.rsplit('/',1)[0] + '/CI_t_less_than_random.csv')
pd.DataFrame(d, columns = idt, index=idt).to_csv(DATAFILES.rsplit('/',1)[0] + '/CI_t_more_than_random.csv')


"""
RESULTS:




Forward direction, self-transitions removed


Transition matrix:

       W         P         A         L
W  0.000000  0.842105  0.131579  0.026316
P  0.383838  0.000000  0.505051  0.111111
A  0.227273  0.272727  0.000000  0.500000
L  0.275000  0.525000  0.200000  0.000000


Confidence that transitions less than Ho:

      W       P       A       L
W  1.0000  0.0000  0.9999  1.0000
P  1.0000  1.0000  0.0000  0.9475
A  1.0000  0.8351  1.0000  0.0000
L  0.9997  0.0010  0.2037  1.0000


Confidence that transitions more than Ho:

      W       P       A       L
W  1.0000  1.0000  0.0001  0.0000
P  0.0000  1.0000  1.0000  0.0525
A  0.0000  0.1649  0.9980  1.0000
L  0.0003  0.9990  0.7965  0.9981



Reverse direction, self-transitions removed

Transition matrix:
       W         P         A         L
W  0.000000  0.593750  0.234375  0.171875
P  0.621359  0.000000  0.174757  0.203883
A  0.147059  0.735294  0.000000  0.117647
L  0.043478  0.239130  0.717391  0.000000


Confidence that transitions less than Ho:
      
      W       P       A       L
W  1.0000  0.0135  0.8358  0.9552
P  0.7171  1.0000  0.7223  0.0853
A  0.9999  0.0001  1.0000  0.6478
L  1.0000  0.8966  0.0000  1.0000

Confidence that transitions more than Ho:

      W       P       A       L
W  1.0000  0.9865  0.1642  0.0449
P  0.2831  1.0000  0.2779  0.9147
A  0.0001  0.9999  0.9990  0.3522
L  0.0000  0.1034  1.0000  0.9990
"""

