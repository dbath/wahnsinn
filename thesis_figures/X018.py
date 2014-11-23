

import os
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pylab as pl




newdf = pd.read_pickle('/groups/dickson/home/bathd/Dropbox/THESIS/X018_adt2_silencing_vs_cells_labelled/mean_CI_and_labelling_adt2_kir.pickle')



def doit():
    plt.scatter(newdf['cells_labelled'], newdf['CI'])    plt.errorbar(newdf['cells_labelled'], newdf['CI'], xerr=newdf['cells_SD'], yerr=newdf['SEM'], linestyle='')
    plt.xlabel('Number of aDT2 cells labelled per hemisphere (mean ' + u"\u00B1" +  'SD)')
    plt.ylabel('Courtship Index '+u"\u00B1"+ 'SEM')



doit()
plt.show()


