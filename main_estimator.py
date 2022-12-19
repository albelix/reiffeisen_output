# Estimator script for prepared data file

"""
The altorithm goes as follows:
1. Loads the prepared individual data file
2. Creates variables for utilities and probabilities from the data
3. Calculates expected utilities (EU) and log-likelihoods (ML)
4. Runs the first ML estimate from prior values based on EU maximization assumption
5. Repeats the iterations for subsequent rounds and collects output
6. Plots utility function and provides outputs: tolerance to chocks, reliability, risk aversion
"""

# 0. Import  required packages and estimation environment (for convenience)
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

"""
1. Loads the prepared data file
"""
rr1 = pd.read_csv("/home/albelix/media/workout.csv").fillna("")  # final
# output

# rr1 = pd.read_csv("/home/albelix/PycharmProjects/reiffeisen_output/workout.csv").fillna("") # final output
rr1.info()
print(rr1.head())

# Makes sure variable types are proper
isinstance(rr1['data'], float)
rr1['data'].dtype
rr1['data'] = pd.to_numeric(rr1['data'])  # .astype(float) - best way to do
rr1['data'] = rr1['data'].astype(float)  # Transform as float
rr1['round'] = pd.to_numeric(rr1['round'])  # .astype(float) - best way to do
rr1['round'] = rr1['round'].astype(int)  # Transform as numeric
rr1['subject'] = 2  # rr1['participant.code'].rank(method='first').astype(int)  ADD TO FIX ID
rr1['subject'] = rr1['subject'].astype(float).astype(int)  # Transform as numeric
rr1['volatility'] = rr1['volatility'].astype(float)

"""
2. Creates variables for utilities and probabilities from the data
"""

# create running max variable
# max of 3 previous price values
rr1['data_max'] = rr1.groupby(["participant.code", "round"])['data'].rolling(3).max().shift(1).reset_index(
    drop=True)
# min of 3 previous price values
rr1['data_min'] = rr1.groupby(["participant.code", "round"])['data'].rolling(3).min().shift(1).reset_index(
    drop=True)
# difference data_max-data_min, past prices range
rr1['data_dif'] = rr1.data_max - rr1.data_min
print(rr1['data_dif'].describe())
# second lag of price
rr1['lag_data'] = rr1.groupby(["participant.code", 'round'])['data'].shift(2).reset_index(drop=True)
# difference between current and lagged price
rr1['delta'] = rr1.groupby(["participant.code", 'round'])['data'].diff().fillna(0).reset_index(drop=True)
# lagged difference between current and lagged price
rr1['lag_delta'] = rr1.groupby(["participant.code"])['delta'].shift(2).reset_index(drop=True)

# rolling 3-period price volatilities
# upper volatility annualized (mohthly - just rescaling)

# range
# maxvol = rr1.groupby(["participant.code","round"]).agg({'volatility': 'min'})
# minvol = rr1.groupby(["participant.code","round"]).agg({'volatility': 'min'})
# range=np.max(maxvol-minvol)
# rr1['ranvol'] = range
# print(rr1[['name', 'ranvol']].describe())

# print(rr1[['name', 'minvol']].describe())


# rr1['rangevol'] = (rr1.groupby(["participant.code","round"])['volatility'].expanding().max()-min().droplevel([0,1]))
# print(rr1[['name', 'rangevol']].describe())

# values - preliminary preparation
# 1 Normalize - needed to redefine lagged variables, perhaps because of "" in lags, instead of nan
rr1.info()
# RoR normalized to 1
rr1['q'] = rr1['data'] / 100  # .div(rr1['endow']) #.div(100)
rr1['q'] = rr1['q'].astype(float)
# Lag 3 data normalized to 1
rr1['lq'] = rr1['lag_data'] / 100
# delta from period to another normalized to one
rr1['dq'] = rr1['delta'] / 100
# lagged delta normalized to 1
rr1['ldq'] = rr1['lag_delta'] / 100
print(rr1[['q', 'lq', 'dq', 'ldq']].describe())

tablout = rr1.loc[:, rr1.columns.isin(list(['round', 'volatility', 'exit.price', 'Index', 'data',
                                            'q', 'lq', 'dq', 'ldq']))].copy()
tablout
print(tablout.describe())

# redefine probs from normalized price q
rr1['upvol'] = (
            rr1['q'] + rr1.groupby(["participant.code", 'round'])['data'].pct_change().rolling(2).std() * (
            252 ** 0.5))  # 1+ was rolling(2)
# lower volatility annualized (mohthly - just rescaling)
rr1['downvol'] = (
            rr1['q'] - rr1.groupby(["participant.code", 'round'])['data'].pct_change().rolling(2).std() * (
            252 ** 0.5))  # 1-
# implied probability up
rr1['probup'] = np.abs(rr1['upvol']) / (np.abs(rr1['upvol']) + np.abs(rr1['downvol']))
# implied probability down
rr1['probdown'] = np.abs(rr1['downvol']) / (np.abs(rr1['upvol']) + np.abs(rr1['downvol']))
print(rr1[['probup', 'probdown']].describe())

# # of times did smt
rr1['numreac'] = (rr1.groupby(["participant.code", "round"])['name'].expanding().count().droplevel(
    [0, 1])) / 24
print(rr1[['name', 'numreac']].describe())

tabout = rr1.loc[:, rr1.columns.isin(list(['round', 'volatility', 'exit.price', 'Index', 'data',
                                           'data_max', 'data_min', 'data_dif', 'lag_data', 'delta',
                                           'lag_delta', 'upvol', 'downvol', 'probup', 'probdown']))].copy()
tabout
print(tabout.describe())

# show decision patterns
# tabout=onsell.pivot_table(values=["volatility"], index=['participant.code'], columns=['round'])
# tabout=tabout.transpose(copy=False)
# tabout

# tablout=pd.crosstab(rr1['volatility'], rr1['exit.price'], rr1['Choice'],aggfunc='count').stack().reset_index(
#     name='round')
# tablout
onsell = rr1[(rr1['Choice'] == 1)]
onesell = onsell.loc[:, onsell.columns.isin(list(['round', 'Index', 'volatility', 'exit.price']))].copy()
onesell
print(onesell.describe())

# Treatment of large chocks: calculate average over previous ranges for this case
rr1['dq_avg'] = rr1.groupby(["participant.code", "round"])['dq'].rolling(2).mean().shift(1).reset_index(
    drop=True)
# utilities of outcomes/r/r
# rr1['xup'] = rr1['q'] + abs(rr1['dq'])
# rr1['xdown'] = rr1['q'] - abs(rr1['dq'])  # pd.Series(np.zeros(rr1.shape[0]))
# rr1['negvol']=rr1['volatility']*-1
# rr1['negvol'] = rr1['negvol'].astype(float)
rr1['xup0'] = rr1['q'] * (1 + round(rr1['volatility'],3))
rr1['xdown0'] = rr1['q'] * (1 - round(rr1['volatility'],3))
def valueup(rr1):
   if rr1['xup0'] >= 1 :
      return rr1['xup0']-1
   if rr1['xup0'] < 1:
      return np.abs(rr1['xup0']-1)
rr1['xup'] = rr1.apply(valueup, axis=1)
def valuedown(rr1):
   if rr1['xdown0'] >= 1 :
      return rr1['xdown0']-1
   if rr1['xdown0'] < 1:
      return np.abs(rr1['xdown0']-1)
rr1['xdown'] = rr1.apply(valuedown, axis=1)
rr1['xupm'] = rr1['xup']
rr1['xdownm'] = rr1['xdown']
rr1.loc[(rr1['xup0'] < 1) & (rr1['xdown0'] < 1), 'xup'] = np.NaN
rr1.loc[(rr1['xup0'] < 1) & (rr1['xdown0'] < 1), 'xdown'] = np.NaN
rr1.loc[(rr1['xup0'] >= 1) | (rr1['xdown0'] >= 1), 'xupm'] = np.NaN
rr1.loc[(rr1['xup0'] >= 1) | (rr1['xdown0'] >= 1), 'xdownm'] = np.NaN


pd.Series(np.zeros(rr1.shape[0]))

# rr1.loc[rr1['delta'] >= -50, 'xdown'] = rr1.q - rr1.dq
# rr1.loc[rr1['delta'] < -50, 'xdown'] = rr1.q - rr1.dq_avg
rdif = rr1['xup'] - rr1['xdown']
print(rdif.describe())

tablarout = rr1.loc[:, rr1.columns.isin(list(['volatility', 'round', 'Index', 'subject', 'data', 'delta',
                                              'q', 'dq', 'dq_avg', 'xup0', 'xup', 'xdown0', 'xdown',
                                              'xupm', 'xdownm',
                                              'delta', 'probup', 'probdown', 'Choice', 'numreac' ]))].copy()

tablarout
print(tablarout.describe())

counts = pd.Series(rdif).value_counts().reset_index().sort_values('index').reset_index(drop=True)
print(counts)

to = tablarout.to_numpy() # convert df to ndarray
long=len(to)  #rows length of array
# newcol=to[to[0:long,10]-1, None]
# to=np.append[to, newcol,1]

"""
3. Calculates expected utilities (EU) and log-likelihoods (ML)
"""
# Expected utility
#rr1['EX'] = (1 - np.exp(-par[2] * tou['xup']))/par[2] * rr1['probup'] + rr1['xdown'] * rr1['probdown']
#print(rr1['EX'].describe())
# calculate log - replace negative values by NA first
# rr1['EX'].loc[(rr1['EX'] < 0)] = 0.00001  # simpler way to replace single negative values
#rr1.loc[rr1['EX'] < 0, 'EX'] = 0.00001  # simpler way to replace single negative values # previous version
# gives copy error

# print(rr1['EX'].describe())

# Log-likelihood and corrections
#rr1['mainlog'] = np.log(rr1['EX'] / rr1['q']).dropna()
#mainU=rr1['mainlog']
#sterreu=np.std(mainU)
# print(rr1['mainlog'].describe())
# rr1['dec0'] = pd.Series(np.zeros(rr1.shape[0]))
# rr1.loc[rr1['mainlog'] >= 0, 'dec0'] = 0  # keep
# rr1.loc[rr1['mainlog'] < 0, 'dec0'] = 1  # sell
# print(rr1[['mainlog', 'dec0', 'Choice']].describe())

print("dataset prepared")

"""
4. Main estimation algorithm: first step
"""


from sklearn import preprocessing
parameters: ndarray = np.array([])

# Individual subject (automatically fulfilled)
indiv = 2

# Selecting data for the chosen subject


touse = rr1[(rr1['subject'] == indiv)]
# tousebase = touss  # Kept for comparison purposes at the end of algorithm

# Selection of needed columns
touss = touse[['volatility', 'round', 'Index', 'subject', 'data', 'delta', 'q', 'dq', 'dq_avg', 'xup0',
              'xdown0', 'delta', 'probup', 'probdown', 'Choice', 'numreac']].astype(np.float64)
touss['Indicator'] = np.where(touss['xup0'] >= 1, 1,0)
touss['Indicator'].astype(bool)
str(touss['Indicator'])

tou = touss[(touss['round'] <= 5)]
# tou = tou[~tou.isin([np.nan, np.inf, -np.inf]).any(1)]
str(tou)
long=len(tou)
"""
4.1. Defining priors
"""
# getting priors
voles = [0.01, 0.05, .10, .15, .20, .25, .30, .35, .40, .45, .50]  # volatilities
#rvec = [3, 2.5, 2, 1.6, 1.2, 1, .8, .6, .4, .2, .05]  # prior risk aversion
rvec = [4.5,3,2.4,1.6,.8,.3,-.3,-.8,-1.6,-2.2,-3]  # prior
# risk aversion
volat = tou.loc[0,'volatility'].copy()
print("getting risk aversion corresponding to prior volatility")
print(volat)
z = voles.index(volat)  # index of prior risk aversion
r0 = rvec[z]
print("prior risk aversion r0 corresponds to chosen volatility")
print(r0)
# tolerance
# tou['belowprice'] = tou[tou['q']<1]
tou['belowprice'] = np.where(tou['q']<1, tou['q'],0).copy()
#tou['meannum'] = tou.groupby(["round"])['q'].mean().reset_index(drop=True)
theta0 = 1-tou['belowprice'].mean()
print("prior tolerance to dropdown is mean duration of holdings (following dropdown)")
print(theta0)
# noise
# tou['EX'] = (1 - np.exp(-r0 * tou['xup']))/r0 * tou['probup'] + (1 - np.exp(-r0 * tou['xup']))/r0 * tou[
#     'probdown']
# tou['mainlog'] = np.log( (tou['EX'] + theta0 *(1-tou['belowprice']))/ (1 - np.exp(-r0 * tou['q']))/r0)
# print(tou[['belowprice', 'EX', 'mainlog', 'numreac']].describe())
# tou['mainlog'].loc[(tou['mainlog'] < 0)] = 0.00001  # correction in case of very high theta
# mainU = tou['mainlog']
# nu0=np.std(mainU)
nu0 = np.std(tou['volatility'])
print("prior stdev is that of non-normalized values of expected utility")
print(nu0)
# loss aversion
lamb0=2

# theta_start = np.array([r0, nu0, theta0])
# params = theta_start
# print(params)

# Omitting theta
theta_start = np.array([r0, nu0, lamb0])
params = theta_start
print(params)


# tou['utmax'] = None
# tou.loc[:,'utmax'] = np.amax(tou['xup']) + 0.001
# tou['utmax']= np.amax(tou['xup']) + 0.01

# This is for utility normalization, atheoretical
#utmax = np.amax(tou['xup']) + 0.01

# this is for loss aversion
# lamb = 1



tou = tou[~tou.isin([np.nan, np.inf, -np.inf]).any(1)]
tou = tou.replace([np.inf, -np.inf], np.nan)
tou['Choice'].describe()
tou=tou.reset_index()

# potential convertion to arrays
# to=tou.to_numpy()
# type(to)

"""
4.2. Define log-likelihood
"""
from sklearn import preprocessing

# Set up ML model
def neg_loglike(params):
#    thet = params[2]
    r = params[0]
    nu = params[1]
    lossa = params[2]

    # lamb = params[3]
    # long=np.count_nonzero(['qto[:,1] == 1)

    # tou['utmax']=np.amax(tou['xup'])+0.01
    # y = (np.log(
    #     (((1 - np.exp(- r * tou['xup'])) / (1 - np.exp(-r * utmax))) * tou['probup'] +
    #      ((1 - np.exp(-r * tou['xdown'])) / (1 - np.exp(-r * utmax))) * tou['probdown']) / tou['q']) -
    #      thet*tou['numreac']) / (np.sqrt(2) * nu) # adjust for utility in 'q'

    # CARA
    # y = (np.log(
    #     (((1 - np.exp(- r * tou['xup'])) / r) * tou['probup'] +  # + lamb *
    #      ((1 - np.exp(- r * tou['xdown'])) / r) * tou['probdown']) / ((1 - np.exp(- r * tou['q'])) / r)) ) / (np.sqrt(2) * nu)
         #- thet / tou['EX'] * (1 - tou['belowprice']) \

    # CRRA
    # y = (np.log(
    #     ((-lossa*tou['xup0']**(1-r))/(1-r) * tou['probup'] +
    #          (-lossa*tou['xdown0']**(1-r)) /(1-r) * tou['probdown'])/
    #          ((tou['q']**(1-r)) / (1-r))) / (np.sqrt(2) * nu))

# works???
    if tou[tou['Indicator']==1].any().any():
        y = (np.log(
            (((tou['xup0']-1) ** (1 - r)) / (1 - r) * tou['probup'] +
             ((tou['xdown0']-1) ** (1 - r)) / (1 - r) * tou['probdown']) /
            (((tou['q']-1) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))
    elif tou[tou['Indicator']==0].any().any():
        y = (np.log(
             ((-lossa * -((1-tou['xup0']) ** (1 - r)) / (1 - r)) * tou['probup'] +
              (-lossa * -((1-tou['xdown0']) ** (1 - r)) / (1 - r)) * tou['probdown']) /
             ((-(1-tou['q']) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))

    # y = np.where(tou[tou['Indicator'] == 1], (np.log(
    #     (np.dot((math.pow(tou['xup'],(1 - r)) / (1 - r), tou['probup']) +
    #      np.dot((math.pow(tou['xdown'],(1 - r)) / (1 - r), tou['probdown'])))) /
    #     (math.pow(tou['q'],(1 - r)) / (1 - r))) / (np.sqrt(2) * nu)),
    #              (np.log(
    #              (np.dot(np.dot(-lossa, math.pow(-tou['xupm'],(1 - r)) / (1 - r)), tou['probup']) +
    #               (np.dot(np.dot(-lossa, math.pow(-tou['xdownm'],(1 - r)) / (1 - r)), tou['probdown']))) /
    #              (math.pow(tou['q'],(1 - r)) / (1 - r))) / (np.sqrt(2) * nu)))


    # y = np.select([tou[tou['Indicator'] == 1], tou[tou['Indicator'] == 0]], [(np.log(
    #         ((tou['xup'] ** (1 - r)) // (1 - r) * tou['probup'] +
    #          (tou['xdown'] ** (1 - r)) // (1 - r) * tou['probdown']) //
    #         ((tou['q'] ** (1 - r)) // (1 - r))) // (np.sqrt(2) * nu)),
    #         (np.log(
    #                  ((-lossa * (-tou['xupm'] ** (1 - r)) // (1 - r)) * tou['probup'] +
    #                   (-lossa * (-tou['xdownm'] ** (1 - r)) // (1 - r)) * tou['probdown']) //
    #                  ((tou['q'] ** (1 - r)) // (1 - r))) // (np.sqrt(2) * nu))]).astype(np.float64)

    # def posit(params):
    #     r = params[0]
    #     nu = params[1]
    #     return (np.log(
    #         ((tou['xup'] ** (1 - r)) / (1 - r) * tou['probup'] +
    #          (tou['xdown'] ** (1 - r)) / (1 - r) * tou['probdown']) /
    #         ((tou['q'] ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))
    #
    # def negat(params):
    #     r = params[0]
    #     nu = params[1]
    #     lossa = params[2]
    #     return (np.log(
    #                  ((-lossa * (-tou['xup'] ** (1 - r)) / (1 - r)) * tou['probup'] +
    #                   (-lossa * (-tou['xdown'] ** (1 - r)) / (1 - r)) * tou['probdown']) /
    #                  ((tou['q'] ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))
    # # tou = tou.to_numpy()
    # y = np.piecewise(tou, [tou[tou['Indicator'] == 1], tou[tou['Indicator'] == 0]],
    #                   [posit(params), negat(params)])
#    return Y

#np.select([tou[tou['xup0'] >= 1], tou[tou['xup0'] < 1]], [
# y = np.where(tou[tou['xup0'] >= 1], (np.log(
    #     (math.pow(tou['xup'], (1 - r)) / (1 - r) * tou['probup'] +
    #      math.pow(tou['xdown'], (1 - r)) / (1 - r) * tou['probdown']) /
    #     (math.pow(tou['q'], (1 - r)) / (1 - r))) / (np.sqrt(2) * nu)),
    #              (np.log(
    #                  ((-lossa * -math.pow(tou['xupm'], (1 - r)) / (1 - r)) * tou['probup'] +
    #                   (-lossa * -math.pow(tou['xdownm'], (1 - r)) / (1 - r)) * tou['probdown']) /
    #                  (math.pow(tou['q'], (1 - r)) / (1 - r))) / (np.sqrt(2) * nu)))

# y = np.where(tou[tou['xup0'] >= 1], (np.log(
#     (np.dot((tou['xup']) ** (1 - r) / (1 - r), tou['probup']) +
#      (np.dot(tou['xdown']) ** (1 - r) / (1 - r)), tou['probdown']) /
#     ((tou['q']) ** (1 - r)) / (1 - r)) / (np.sqrt(2) * nu)),
#              (np.log(
#                  (np.dot(-lossa, -np.dot((tou['xupm']) ** (1 - r) / (1 - r)), tou['probup']) +
#                   (np.dot(-lossa, -np.dot(tou['xdownm'] ** (1 - r) / (1 - r), tou['probdown'])))) /
#                  ((tou['q']) ** (1 - r)) / (1 - r)) / (np.sqrt(2) * nu)))

# if tou[tou['xup0'] >=1 ].bool():
    #    y = (np.log(
    #          (((tou['xup'])**(1-r) / (1-r) * tou['probup'] +
    #           ((tou['xdown'])**(1-r)) /(1-r) ) * tou['[probdown'])/
    #          ((tou['q'])**(1-r)) / (1-r))) / (np.sqrt(2) * nu)
    # elif tou[tou['xup0']<1].bool():
    #     y = (np.log(
    #         ((-lossa*-(tou['xupm']) ** (1 - r) / (1 - r) * tou['probup'] +
    #           ((-lossa*-tou['xdownm']) ** (1 - r)) / (1 - r)) * tou['[probdown']) /
    #         ((tou['q']) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu)
#    return y
    # y = np.piecewise(to, [to[:,13]>0, to[:,17]>0],
    #              [(np.log(
    #                  (((to[:,13])**(1-r) / (1-r) * tou[:, 9] +
    #                   ((tou[:,14])**(1-r)) /(1-r) ) * tou[:,10])/
    #                  ((tou[:,7])**(1-r)) / (1-r))) / (np.sqrt(2) * nu),
    #               (np.log(
    #                   -lossa * -(((to[:, 17]) ** (1 - r) / (1 - r) * tou[:, 9] +
    #                     -lossa * -((tou[:, 18]) ** (1 - r)) / (1 - r)) * tou[:, 10]) /
    #                   ((tou[:, 7]) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu)
    #               ])


                 # y=[(np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
                 #            ((1 - np.exp(- r * to[0:long,7])) / r) * to[0:long,9]) / to[0:long,10]) - thet *
#                     to[0:long,16]) / (np.sqrt(2) * nu),
#                   (np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
#                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
#                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu),
#                   (np.log((lamb * ((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
#                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
#                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu)])

# y = (np.log(
    #     (((tou['xup']) ** (1 - r) / (1 - r) * tou['probup'] +
    #       ((tou['xdown']) ** (1 - r)) / (1 - r)) * tou['probdown']) /
    #     ((tou['q']) ** (1 - r)) / (1 - r))) + thet / tou['EX'] * (1 - tou['belowprice']) / (
    #             np.sqrt(2) * nu)
    # y = (np.log(
    #     (((tou['xup'])**(1-r) / (1-r) * tou['probup'] +
    #      ((tou['xdown'])**(1-r)) /(1-r) ) * tou['probdown']) * (1 + thet/tou['EX'] * (1 - tou['belowprice'])) /
    #     ((tou['q'])**(1-r)) / (1-r))) / (np.sqrt(2) * nu)
    # y = (np.log(
    #     (((1 - np.exp(- r * tou['xup']))/r) * tou['probup'] +  # + lamb *
    #      ((1 - np.exp(- r * tou['xdown']))/r) * tou['probdown'])*(1 + thet * (1-tou['belowprice']) )/
    #      (1 - np.exp(- r * tou['q']))/r)) / (np.sqrt(2) * nu)
    # y = np.piecewise(to, [(to['xup'] > 1) & (to['xdown'] > 1), (to['xup'] > 1) & (to['xdown'] <= 1),
    #                        (to['xup'] <= 1) & (to['xdown'] <= 1)],
    #                  [(np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             ((1 - np.exp(- r * to['xdown'])) / r) * to['probdown']) / to['q']) - thet *
    #                     to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu)])

    # y = np.piecewise(to, [(to[0:long,6] > 1) & (to[0:long,7] > 1), (to[0:long,6] > 1) & (to[0:long,7] <= 1),
    #                        (to[0:long,6] <= 1) & (to[0:long,7] <= 1)],
    #                  [(np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             ((1 - np.exp(- r * to[0:long,7])) / r) * to[0:long,9]) / to[0:long,10]) - thet *
    #                     to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu)])
    #
    # def piecewise_linear(x, x0, x1, b, k1, k2, k3):
    #     condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    #     funclist = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0),
    #                 lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1)]
    #     return np.piecewise(x, condlist, funclist)
    #
    # p, e = optimize.curve_fit(piecewise_linear, x, y)
    #
    #                  [(np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             ((1 - np.exp(- r * to['xdown'])) / r) * to['probdown']) / to['q']) - thet *
    #                     to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu)])

    # y = np.piecewise(to, [(to[0:long,6] > 1) & (to[0:long,7] > 1), (to[0:long,6] > 1) & (to[0:long,7] <= 1),
    #                        (to[0:long,6] <= 1) & (to[0:long,7] <= 1)],
    #                  [(np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             ((1 - np.exp(- r * to[0:long,7])) / r) * to[0:long,9]) / to[0:long,10]) - thet *
    #                     to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu)])
    #
    # def piecewise_linear(x, x0, x1, b, k1, k2, k3):
    #     condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    #     funclist = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0),
    #                 lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1)]
    #     return np.piecewise(x, condlist, funclist)
    #
    # p, e = optimize.curve_fit(piecewise_linear, x, y)
    #
    #                        (to['xup'] <= 1) & (to['xdown'] <= 1)],
    #                  [(np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             ((1 - np.exp(- r * to['xdown'])) / r) * to['probdown']) / to['q']) - thet *
    #                     to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to['xup'])) / r) * to['probup'] +
    #                             (lamb * ((1 - np.exp(- r * to['xdown'])) / r)) * to['probdown']) / to[
    #                                'q']) - thet * to['numreac']) / (np.sqrt(2) * nu)])

    # y = np.piecewise(to, [(to[0:long,6] > 1) & (to[0:long,7] > 1), (to[0:long,6] > 1) & (to[0:long,7] <= 1),
    #                        (to[0:long,6] <= 1) & (to[0:long,7] <= 1)],
    #                  [(np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             ((1 - np.exp(- r * to[0:long,7])) / r) * to[0:long,9]) / to[0:long,10]) - thet *
    #                     to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu),
    #                   (np.log((lamb * ((1 - np.exp(- r * to[0:long,6])) / r) * to[0:long,8] +
    #                             (lamb * ((1 - np.exp(- r * to[0:long,7])) / r)) * to[0:long,9]) / to[
    #                                0:long,10]) - thet * to[0:long,16]) / (np.sqrt(2) * nu)])
    #
    # def piecewise_linear(x, x0, x1, b, k1, k2, k3):
    #     condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    #     funclist = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0),
    #                 lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1)]
    #     return np.piecewise(x, condlist, funclist)
    #
    # p, e = optimize.curve_fit(piecewise_linear, x, y)
    #
                      #           ((1 - np.exp(- r * tou['xdown'])) / r) * tou['probdown']) / tou['q']) - thet *
                      #   tou['numreac']) / (np.sqrt(2) * nu),
                      # (np.log((((1 - np.exp(- r * tou['xup'])) / r) * tou['probup'] +
                      #           (lamb * ((1 - np.exp(- r * tou['xdown'])) / r)) * tou['probdown']) / tou[
                      #              'q']) - thet * tou['numreac']) / (np.sqrt(2) * nu),
                      # (np.log((lamb * ((1 - np.exp(- r * tou['xup'])) / r) * tou['probup'] +
                      #           (lamb * ((1 - np.exp(- r * tou['xdown'])) / r)) * tou['probdown']) / tou[
                      #              'q']) - thet * tou['numreac']) / (np.sqrt(2) * nu)])
    #
    # def piecewise_linear(x, x0, x1, b, k1, k2, k3):
    #     condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    #     funclist = [lambda x: k1 * x + b, lambda x: k1 * x + b + k2 * (x - x0),
    #                 lambda x: k1 * x + b + k2 * (x - x0) + k3 * (x - x1)]
    #     return np.piecewise(x, condlist, funclist)
    #
    # p, e = optimize.curve_fit(piecewise_linear, x, y)
    #
    # forget about lambda
    #     (((1 - np.exp(- r * tou['xup'])) / (1 - np.exp(-r * utmax))) * tou['probup'] + lamb * (
    #             (1 - np.exp(-r * tou['xdown'])) / (1 - np.exp(-r * utmax))) * tou['probdown']) / tou['q']) - thet ) / (np.sqrt(2) * nu) # adjust for utility in 'q'
# def fullneg_loglike(Y):
    #    mu = np.log(norm.cdf(Y) * tou['dec0'] + (1 - tou['dec0']) * (1 - norm.cdf(Y)))
    Y = preprocessing.scale(y)  # standardize data
    mu = np.log(norm.cdf(Y)) * tou['Choice'] + (1 - tou['Choice']) * np.log(1 - norm.cdf(Y))
    return -1 * mu.sum()

print("defined")
# Optimize the log-likelihood over our parameters using minimize from python's scipy.optimize package:

# Define prior volatility of number of sales
# utilities of outcomes
# ML Maximization, first iteration
theta_start = np.array([r0, nu0, lamb0]) # theta0,
params = theta_start
res = minimize(neg_loglike, theta_start, method='L-BFGS-B', options={'disp': True})
# print(res)
# print(res['x'])
par = res['x']
parameters = np.append(parameters, par)
print(parameters)

"""
5. Repeats the iterations for subsequent rounds and collects output
"""
for iter in range(6, 11):
    # Step n: take values from previous round's estimates
    tou1 = touss[(touss['round'] <= iter)]
#    tou = touss[(touss['subject'] == indiv) & (touss['round'] <= iter)]  # select data
    # tou = touss[['volatility', 'round', 'Index', 'subject', 'data', 'delta', 'xup', 'xdown', 'probup',
    #              'probdown', 'q', 'dq',  'Choice', 'numreac', 'belowprice', 'EX', 'mainlog']]
    tou1 = tou1[~tou1.isin([np.nan, np.inf, -np.inf]).any(1)]
    tou1 = tou1.replace([np.inf, -np.inf], np.nan)
    # utmax = np.amax(tou['xup']) + 0.01

    #    lamb = 1
    # print("utilities")
    # print( (1 - np.exp(-r0 * tou['xup']))/r0 )
    # print( (1 - np.exp(-r0 * tou['xdown']))/r0 )

#    touse['EX'] = (1 - np.exp(-par[2] * touse['xup'])) / par[2]* touse['probup'] + (1 - np.exp(-par[2] *
#     touse['EX'] = (((touse['xup']) ** (1 - par[0]) / (1 -  par[0]) * touse['probup'] +
#               ((touse['xdown']) ** (1 -  par[0])) / (1 -  par[0])) * touse['probdown'])
#     touse['mainlog'] = np.log(
#         (touse['EX'] + par[0] * (1 - touse['belowprice'])) / (1 - np.exp(-par[0] * touse['q'])) / par[1])
#     print(touse[['belowprice', 'EX', 'mainlog', 'numreac']].describe())


    # tou['EX'] =  (1 - np.exp(-par[2] * tou['xup']))/par[2]  * tou['probup'] + (1 - np.exp(-par[2] * tou['xup']))/par[2] * tou['probdown']
    # tou['mainlog'] = np.log(
    #     (tou['EX'] - par[0])/ tou['q'])  # ((1 - np.exp(-r0 * tou['q'])) / (1 - np.exp(-r0 * utmax))) )
    # print("mainlog")
    # mainU=tou['mainlog']
    # par[1]=np.std(mainU)
    # print(mainU)
    # tou['dec0'] = pd.Series(np.zeros(tou.shape[0]))
    # tou.loc[tou['mainlog'] >= 0, 'dec0'] = 0  # keep
    # tou.loc[tou['mainlog'] < 0, 'dec0'] = 1  # sell
    # tou['dec1'] = np.maximum(tou['dec0'], tou['Choice'])
    # print(tou[['mainlog', 'dec0', 'Choice', 'dec1']].describe())

    # touse = tou[
    #     ["subject", "round", "Index", "xup", "xdown", "q", "dq", "probup", "probdown", "mainlog",
    #      "Choice", "EX", "dec0"]]


    #  Set up ML model for iteration
    def neg_loglikelihood(params):
        # thet = params[2]
        nu = params[1]
        r = params[0]
        lossa = params[2]

        #tou['utmax'] = np.amax(tou['xup']) + 0.1
        print("r")
        print(r)
        # print((1 - np.exp(-r0 * tou['xup'])))
        # print((1 - np.exp(-r0 * utmax)))
        # y = (np.log(
        #     (((1 - np.exp(- r * tou['xup'])) / (1 - np.exp(-r * utmax))) * tou['probup'] + # + lamb *
        #          ((1 - np.exp(-r * tou['xdown'])) / (1 - np.exp(-r * utmax))) * tou[
        #          'probdown']) / tou['q']) - thet*tou['numreac']) / (np.sqrt(2) * nu)
        # # cara
        # y = (np.log(
        #     (((1 - np.exp(- r * touse['xup'])) / r) * touse['probup'] +  # + lamb *
        #      ((1 - np.exp(- r * touse['xdown'])) / r) * touse['probdown']) / ((1 - np.exp(- r * touse[
        #         'q'])) / r)) ) / (np.sqrt(2) * nu)
            # - thet/touse['EX'] * (1 - touse['belowprice'])
        # CRRA
        # y = (np.log(
        #     (((touse['xup']) ** (1 - r) / (1 - r) * touse['probup'] +
        #       ((touse['xdown']) ** (1 - r)) / (1 - r)) * touse['probdown'])  /
        #     ((touse['q']) ** (1 - r)) / (1 - r))) + thet/touse['EX'] * (1 - touse['belowprice']) / (
        # np.sqrt(2) * nu)
        if tou1[tou1['Indicator']==1].any().any():
            y = (np.log(
                (((tou1['xup0']-1) ** (1 - r)) / (1 - r) * tou1['probup'] +
                 ((tou1['xdown0']-1) ** (1 - r)) / (1 - r) * tou1['probdown']) /
                (((tou1['q']-1) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))
        elif tou1[tou1['Indicator']==0].any().any():
            y = (np.log(
                ((-lossa * -((1 - tou1['xup0']) ** (1 - r)) / (1 - r)) * tou1['probup'] +
                 (-lossa * -((1 - tou1['xdown0']) ** (1 - r)) / (1 - r)) * tou1['probdown']) /
                ((-(1 - tou['q']) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu))

        Y = preprocessing.scale(y)
        mu = np.log(norm.cdf(Y)) * tou1['Choice'] + (1 - tou1['Choice']) * np.log(1 - norm.cdf(Y))
        #        mu = np.log(norm.cdf(Y) * tou['dec1'] + (1 - tou['dec1']) * (1 - norm.cdf(Y)))
        return -1 * mu.sum()


    # ML Maximization for iteration
#    touse = touse[~touse.isin([np.nan, np.inf, -np.inf]).any(1)]
    param_start = np.array([par])
    print('prior vector')
    print(theta_start)
    res = minimize(neg_loglikelihood, param_start, method='L-BFGS-B', options={'disp': True})  # BFGS
    # Nelder-Mead
    par = res['x']
    # parameters = np.vstack([parameters, par])
    print('iteration no ')
    print(iter)
    print(parameters)

"""
6. Output
"""

x = np.linspace(0.01, 2, 200)
rpar = par[0]
# CARA
#y = (1 - np.exp(-rpar * x)) / rpar
#CRRA
y = x**(1-rpar) / (1-rpar)
plt.plot(x, y)
plt.show()

#print("Coefficient of risk aversion CARA (r<0 risk seeking, r>0 risk averse)")
print("Coefficient of risk aversion CRRA (r<0 risk seeking, r>0 risk averse)")
print(par[0].round(2))

print("Reliability of estimate (the closer to 0 the better) ")
print(par[1].round(2))

#print("Tolerance to chocks (the lower is estimate <0 the higher is tolerance")
print("Loss aversion (the larger is estimate >0 the higher is loss aversion")
print(par[2].round(2))



# volatilities and dropdowns by this individual
print(onesell)

# print(tousebase['volatility'].value_counts())
# print(tousebase.groupby(['round']).size())
