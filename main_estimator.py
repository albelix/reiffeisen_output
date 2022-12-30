# Estimator script for prepared data file

"""
The altorithm goes as follows:
0. Import packages and parameters
1. Loads the prepared individual data file
2. Creates variables for utilities and probabilities from the data
3. Construct additional variables and prepare dataset
4. Define priors
5. Runs the first ML estimate from prior values based on EU maximization assumption
6. Repeats the iterations for subsequent rounds and collects output
7. Plots utility function and provides outputs: risk aversion, estimation precision,
loss aversion (not estimated), tolerance to dropdowns
"""
"""
0. Import  required packages and estimation environment
"""
from numpy import ndarray

import numpy as np
import pandas as pd
from scipy.stats import norm
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
rr1 = pd.read_csv("/home/albelix/media/workout.csv").fillna("")
rr1.info()
print(rr1.head())

# Makes sure variable types are proper
isinstance(rr1['data'], float)
rr1['data'].dtype
rr1['data'] = rr1['data'].astype(float)  # Transform as float
rr1['round'] = rr1['round'].astype(int)  # Transform as numeric
rr1['subject'] = 2  # to ensure values are non-conflicting
rr1['subject'] = rr1['subject'].astype(float).astype(int)  # Transform as numeric
rr1['volatility'] = rr1['volatility'].astype(float)

"""
2. Creates variables for utilities and probabilities from the data
"""
# values - preliminary preparation
# Normalized rate of return to 1
rr1['q'] = rr1['data'] / 100
rr1['q'] = rr1['q'].astype(float)

# redefine probabilities from normalized price q: upper volatility
rr1['upvol'] = (
            rr1['q'] + rr1.groupby(["participant.code", 'round'])['data'].pct_change().rolling(2).std() * (
            252 ** 0.5))  # lower volatility annualized (mohthly - just rescaling)
rr1['downvol'] = (
            rr1['q'] - rr1.groupby(["participant.code", 'round'])['data'].pct_change().rolling(2).std() * (
            252 ** 0.5))
# implied probability up
rr1['probup'] = np.abs(rr1['upvol']) / (np.abs(rr1['upvol']) + np.abs(rr1['downvol']))
# implied probability down
rr1['probdown'] = np.abs(rr1['downvol']) / (np.abs(rr1['upvol']) + np.abs(rr1['downvol']))
print(rr1[['probup', 'probdown']].describe())

# Summary table for portfolio values at termination
onsell = rr1[(rr1['Choice'] == 1)]
onesell = onsell.loc[:, onsell.columns.isin(list(['round', 'Index', 'volatility', 'exit.price']))].copy()
onesell
print(onesell.describe())

# Higher and lower outcomes for lotteries
rr1['xup0'] = rr1['q'] * (1 + round(rr1['volatility'],3))
rr1['xdown0'] = rr1['q'] * (1 - round(rr1['volatility'],3))

print("dataset prepared")

"""
3. Construct additional variables and prepare dataset
"""
from sklearn import preprocessing
parameters: ndarray = np.array([])
# Individual subject (dummy for the code)
indiv = 2
# Selecting data for the chosen subject

touse = rr1[(rr1['subject'] == indiv)]

# Selection of needed columns
touss = touse[['volatility', 'round', 'Index', 'subject', 'data', 'q', 'xup0',  #'dq', 'dq_avg', 'xup0',
              'xdown0',  'probup', 'probdown', 'Choice']].astype(np.float64)
# Indicator of positive rate of return
touss['Indicator'] = np.where(touss['xup0'] >= 1, 1,0)
touss['Indicator'].astype(bool)
str(touss['Indicator'])

# Select sample for prior estimate
tou = touss[(touss['round'] <= 5)]
str(tou)
long=len(tou)

vardec=touss['volatility'].std()



"""
4. Define priors
"""
# getting priors
voles = [0.01, 0.05, .10, .15, .20, .25, .30, .35, .40, .45, .50]  # volatilities
rvec = [4.5,3,2.4,1.6,.8,.3,-.3,-.8,-1.6,-2.2,-3]  # prior risk aversion: needs to be scaled for application
# parameter 0: risk aversion
volat = tou.loc[0,'volatility'].copy()
print("getting risk aversion corresponding to prior volatility")
print(volat)
z = voles.index(volat)  # index of prior risk aversion label in range
r0 = rvec[z]
print("prior risk aversion r0 corresponds to chosen volatility")
print(r0)
# parameter 1: noise
nu0 = np.std(tou['volatility'])
print("prior stdev is that of non-normalized values of expected utility")
print(nu0)
# parameter 2: loss aversion (not estimated for parsimony)
lamb0=2
# parameter 3: tolerance to dropdowns
# tou['belowprice'] = tou[tou['q']<1]
tou['belowprice'] = np.where(tou['q']<1, tou['q'],0.001).copy()
theta0 = 1-tou['belowprice'].mean()
print("prior tolerance to dropdown is mean duration of holdings (following dropdown)")
print(theta0)

# prior vector
theta_start = np.array([r0, nu0, lamb0, theta0])
params = theta_start
print(params)

# restrict data to make sure all numbers are as requested for estimation (e.g. no negatives in
# logs)
tou = tou[~tou.isin([np.nan, np.inf, -np.inf]).any(1)]
tou = tou.replace([np.inf, -np.inf], np.nan)
tou['Choice'].describe()
tou=tou.reset_index()

"""
5. Define log-likelihood for 4 parameters: CRRA risk aversion, decision noise, loss aversion (not 
estimated) and tolerance to dropdowns
"""
from sklearn import preprocessing

# Set up ML model
def neg_loglike(params):
    r = params[0]
    nu = params[1]
    lossa = params[2]
    theta = params[3]

    if tou[tou['Indicator']==1].any().any():
        y = (np.log(
            (((tou['xup0']-1) ** (1 - r)) / (1 - r) * tou['probup'] +
             ((tou['xdown0']-1) ** (1 - r)) / (1 - r) * tou['probdown']) /
            (((tou['q']-1) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu) - theta)
    elif tou[tou['Indicator']==0].any().any():
        y = (np.log(
             ((-lossa * -((1-tou['xup0']) ** (1 - r)) / (1 - r)) * tou['probup'] +
              (-lossa * -((1-tou['xdown0']) ** (1 - r)) / (1 - r)) * tou['probdown']) /
             ((-(1-tou['q']) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu)-theta)

    Y = preprocessing.scale(y)  # standardize data
    mu = np.log(norm.cdf(Y)) * tou['Choice'] + (1 - tou['Choice']) * np.log(1 - norm.cdf(Y))
    return -1 * mu.sum()
print("Log likelihood defined")

# Define prior volatility of number of sales
# utilities of outcomes
# ML Maximization, first iteration [r0, nu0, lamb0, thet0])
theta_start = np.array([r0, nu0, lamb0, theta0]) # theta0,
params = theta_start
res = minimize(neg_loglike, theta_start, method='L-BFGS-B', options={'disp': True})
par = res['x']
print(par)
parameters = np.append(parameters, par) # accumulate parameters over iterations
print(parameters)

"""
6. Main run: repeat iterations for subsequent rounds taking values from previous estimates and 
collect output
"""
for iter in range(6, 11):
    tou1 = touss[(touss['round'] <= iter)]
    tou1 = tou1[~tou1.isin([np.nan, np.inf, -np.inf]).any(1)]
    tou1 = tou1.replace([np.inf, -np.inf], np.nan)

    #  Set up ML model for iteration
    def neg_loglikelihood(params):
        r = par[0]
        nu = par[1]
        lossa = par[2]
        theta = par[3]

        if tou1[tou1['Indicator']==1].any().any():
            y = (np.log(
                (((tou1['xup0']-1) ** (1 - r)) / (1 - r) * tou1['probup'] +
                 ((tou1['xdown0']-1) ** (1 - r)) / (1 - r) * tou1['probdown']) /
                (((tou1['q']-1) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu) - theta)
        elif tou1[tou1['Indicator']==0].any().any():
            y = (np.log(
                ((-lossa * -((1 - tou1['xup0']) ** (1 - r)) / (1 - r)) * tou1['probup'] +
                 (-lossa * -((1 - tou1['xdown0']) ** (1 - r)) / (1 - r)) * tou1['probdown']) /
                ((-(1 - tou['q']) ** (1 - r)) / (1 - r))) / (np.sqrt(2) * nu) - theta)

        Y = preprocessing.scale(y)
        mu = np.log(norm.cdf(Y)) * tou1['Choice'] + (1 - tou1['Choice']) * np.log(1 - norm.cdf(Y))
        return -1 * mu.sum()
print("Log-likelihood prepared")

# ML Maximization for iteration
param_start = np.array([par])
print('prior vector')
print(theta_start)
res = minimize(neg_loglikelihood, param_start, method='L-BFGS-B', options={'disp': True})  # BFGS
# Nelder-Mead
par = res['x']
print('iteration no ')
print(iter)
print(par)
parameters = np.append(parameters, par)
print(parameters)

"""
7. Output
"""

x = np.linspace(0.01, 2, 200)
rpar = par[0]
# CARA
#y = (1 - np.exp(-rpar * x)) / rpar
#CRRA
y = np.where(np.abs(par[0]-1)>0.15, x**(1-par[0]) / (1-par[0]), np.log(x))
plt.plot(x, y)
plt.show()

#print("Coefficient of risk aversion CARA (r<0 risk seeking, r>0 risk averse)")
print("Coefficient of risk aversion CRRA (r<0 risk seeking, r>0 risk averse)")
print(par[0].round(11))

print("Variance of choices (closer to 0 = more stabler) ")
print(par[1].round(11))

#print("Tolerance to chocks (the lower is estimate <0 the higher is tolerance")
print("Loss aversion (the larger is estimate >0 the higher is loss aversion")
print(par[2].round(11))

print("Tolerance to dropdowns (the larger the lower)")
print(par[3].round(11))

# Statistics of decisions for this individual
print(onesell)
print( "Warning: no variance in subject's decisions, corner estimates" if vardec<=0.001 else "Estimation "
                                                                                        "complete" )

