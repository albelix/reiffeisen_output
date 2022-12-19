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
import math
from decimal import Decimal
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

sns.set_style(
    style='darkgrid',
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

plt.rcParams['figure.figsize'] = [10, 7]


x = np.linspace(0.01, 2, 200)
volat = [0.01, 0.05, .10, .15, .20, .25, .30, .35, .40, .45, .50]  # volatilities
apar =  [5.00, 3.5,  2.0, 1.0, 0.5, 0.2, -0.01, -0.05, -0.08, -0.1, -0.15]
rpar =  [0.85, 0.7, 0.55, 0.4, 0.25, 0.05, -.1, -0.9, -1.5, -2, -3]



# CARA
#y = (1 - np.exp(-rpar * x)) / rpar
#CRRA
y0 = x**(1-rpar[0]) / (1-rpar[0])
y1 = x**(1-rpar[1]) / (1-rpar[1])
y2 = x**(1-rpar[2]) / (1-rpar[2])
y3 = x**(1-rpar[3]) / (1-rpar[3])
y4 = x**(1-rpar[4]) / (1-rpar[4])
y5 = x**(1-rpar[5]) / (1-rpar[5])
y6 = x**(1-rpar[6]) / (1-rpar[6])
y7 = x**(1-rpar[7]) / (1-rpar[7])
y8 = x**(1-rpar[8]) / (1-rpar[8])
y9 = x**(1-rpar[9]) / (1-rpar[9])
y10 = x**(1-rpar[10]) / (1-rpar[10])

f = plt.figure(1)
plt.plot(x, y0)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)
plt.plot(x, y7)
plt.plot(x, y8)
plt.plot(x, y9)
plt.plot(x, y10)
f.show()

# CARA
Y0 = (1 - np.exp(-apar[0] * x)) / apar[0]
Y1 = (1 - np.exp(-apar[1] * x)) / apar[1]
Y2 = (1 - np.exp(-apar[2] * x)) / apar[2]
Y3 = (1 - np.exp(-apar[3] * x)) / apar[3]
Y4 = (1 - np.exp(-apar[4] * x)) / apar[4]
Y5 = (1 - np.exp(-apar[5] * x)) / apar[5]
Y6 = (1 - np.exp(-apar[6] * x)) / apar[6]
Y7 = (1 - np.exp(-apar[7] * x)) / apar[7]
Y8 = (1 - np.exp(-apar[8] * x)) / apar[8]
Y9 = (1 - np.exp(-apar[9] * x)) / apar[9]
Y10 = (1 - np.exp(-apar[10] * x)) / apar[10]

g = plt.figure(2)
plt.plot(x, Y0)
plt.plot(x, Y1)
plt.plot(x, Y2)
plt.plot(x, Y3)
plt.plot(x, Y4)
plt.plot(x, Y5)
plt.plot(x, Y6)
plt.plot(x, Y7)
plt.plot(x, Y8)
plt.plot(x, Y9)
plt.plot(x, Y10)
g.show()
input()
