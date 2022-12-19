from numpy import ndarray

import numpy as np
import pandas as pd
import scipy.special as sp
import pymc3 as pm
import arviz as az
import numdifftools as ndt
from scipy.stats import norm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

plt.rcParams['figure.figsize'] = [10, 7]

"""
1. Loads the prepared data file
"""
rstat = pd.read_csv("/home/albelix/Documents/Reiffeisen/workout.csv").fillna("")  # final


# summarise volatilities
volat = rstat.groupby(['volatility', 'name'])['data'].std()
volat


to[:,1]

y = np.piecewise(to, [to[:, 13] > 0, to[:, 17] > 0],  [to[:, 1], to[:,2]])