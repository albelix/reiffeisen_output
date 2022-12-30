# Python script to extract subjects decisions data.

"""
The altorithm goes as follows:
1. Get data from original data file (data_2021-11-09_export.csv, for example)
2. Delete unneeded variables.
3. Get data for single subject, and
4. Reshape (expand) prices at the end of each batch of 10 to long format of 2500 obs. per participant.

5. Importing events data (events_11_09_2021_21_07_19_export.csv, for example)
6. Converting all merger variables to the same type and
7. Merging to the main workfile,
8. Defining conditions for dropdown and dropping data which are never observable because of that, and
9. Saving the working file as workout.csv
"""

# 0. import modules
import pandas as pd
import numpy as np
from ast import literal_eval
import itertools
import ast
import gc
import tracemalloc
import matplotlib.pyplot as plt
import multiprocessing, random, sys, os, time

import json
from pandas.io.json import json_normalize
#print(pd.__version__)

# make columns display wide for convenience
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# trace memory allocation
tracemalloc.start()

"""
1. import the main data file, from app log (instead of 'data_2021-11-09_export.csv', insert correct name 
and address)
    The main data file contains 
        unique subject id (variable "participant_code"),
        all price series ("data"), 
        round number ("round"), 
        volatilities selected in each round ("volatility")
        selling or end of the round price ("exit_price") 
    plus some other variables that we need to drop first. 
    Adjust the variables list as needed.
    Insert the number of participant at line 127
"""
#reifdat_df = pd.read_csv("/home/albelix/media/data_2021-11-09_export.csv")
reifdat_df = pd.read_csv("/home/albelix/Documents/Reiffeisen/data_2021-11-09_export.csv")


"""
2. drop inneeded columns by numbers
"""
# print column names and numbers
colist = reifdat_df.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)

print(colist)
cols = [0,2,3,4,5,6,7,9,10,11,14,15,16,17,18,19,20,
        22,23,24,25,27,29,31,32,33,34,35,
        36,37,38,39,41,43,45,46,47,48,49,
        50,51,52,53,55,57,59,60,61,62,63,
        64,65,66,67,69,71,73,74,75,76,77,
        78,79,80,81,83,85,87,88,89,90,91,
        92,93,94,95,97,99,101,102,103,104,105,
        106,107,108,109,111,113,115,116,117,118,119,
        120,121,122,123,125,127,129,130,131,132,133,
        134,135,136,137,139,141,143,144,145,146,147,
        148,149,150,151,153,155,159,160,161,163]
datdf=reifdat_df.drop(reifdat_df.columns[cols],axis=1)
colist = datdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)

datdf=datdf.iloc[:,0:36] # drop further unneeded columns
colist = datdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)

print(datdf.head(10))
print("dataframe created")

# create index for id value
print(type(datdf['participant.code']))
datdf['participant.code'] = datdf['participant.code'].astype(str)
datdf['subject'] = datdf['participant.code'].rank(method='first').astype(int)

# replace unneeded prefixes in names
datdf.columns = datdf.columns.str.replace(".player", "")
datdf.columns = datdf.columns.str.replace("backend.", "")
datdf.columns = datdf.columns.str.replace("reif_survey_0.1", "surv")
for col in datdf.columns:
    print(col)
print(datdf.head(10))
print("all prepared")


"""
3. Explode price lists to rows for a single participant: prepare data
"""

colist = datdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)
shapshot1 = tracemalloc.take_snapshot()
top_stats = shapshot1.statistics('lineno')

"""
4. exploding (reshaping) cycle
    This cycle pick up by rounds three variables in wide format: 
        "volatility" (fixed), 
        "exit price" (end of period price, fixed) and 
        "data" (current price, varying)
    Expand them and the rest of variables in long form, and 
    stacks observations for each round over each other.
"""

dfin = np.empty((2500, 10), dtype=object)   # create empty df to collect reshaped data
d = datdf[datdf['subject'] == 1] # select data for single participant
for count in range(1,11,1):  # create main cycle over rounds
    colfull = [0, 1, 2, 3, 4, 4+(count-1)*3+1, 4+(count-1)*3+2, 4+(count-1)*3+3]  # Pick up three variables in wide format
    print(colfull)
    colist = datdf.columns.to_list()
    dtm = d.iloc[:, colfull] # select colums by numbers and rename them to final names
    dtm = dtm.rename(columns={dtm.columns[3]: 'compnumber'})
    dtm = dtm.rename(columns={dtm.columns[4]: 'buckser'})
    dtm = dtm.rename(columns={dtm.columns[5]: 'volatility'})
    dtm = dtm.rename(columns={dtm.columns[6]: 'data'})
    dtm = dtm.rename(columns={dtm.columns[7]: 'exit.price'})
    dtm['round'] = count
    print(dtm['round'])
    hdr = dtm.columns.values.tolist()
    # reshape and collect price values (data) per batch to a single expanded row
    dtm['data'] = dtm['data'].apply(ast.literal_eval)  # read json object as list
    dtnew = dtm.explode('data', ignore_index=False).reset_index()
    dtn = dtnew.values.tolist()
    # stack values for rounds 1:10
    if count==1:
        dfin[:250]=dtn
    else:
        dfin[250*(count-1):250*count]=dtn
        del [[dtm, dtnew]]  # clear temporary variables for future reuse
        gc.collect()
        dtm = pd.DataFrame()
        dtnew = pd.DataFrame()
    hdr.insert(0,'obsno')
    shapshot2 = tracemalloc.take_snapshot()
    top_stats = shapshot2.statistics('lineno')
    # speed of processing report
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

print("reached end of cycle")
dfsubj = pd.DataFrame(dfin, columns=hdr)
print(type(dfsubj))
# creating indices
dfsubj['TotalIndex'] = np.arange(len(dfsubj)) # total index of observation per participant, python format
dfsubj["Index"] = dfsubj.groupby("round")["TotalIndex"].rank(method="first", ascending=True) # observation index per round
dfsubj["Index"] = dfsubj.groupby("round")["TotalIndex"].rank(method="first", ascending=True) # observation index per round
# savind dataframe
dfsubj.to_csv('data_subject.csv', index=False)
getpartcode=dfsubj['participant.code'][0] # get label (code) of the current participant - JUST CHANGE THIS
print("decisions database completed")
type(dfsubj['Index'])


# summary statistics
print(dfsubj['Index'].value_counts()) # distribution
print(dfsubj['Index'].describe()) # summary statistics
print (dfsubj['Index'].apply(type)) # var type



"""
5. Calling event data
    This data file contains participant_code (variable to merge on) plus
        events names ("name") for volatility choice (prefix "Decision") and trading decisions (prefix "Trade") 
            decisions in trading periods are surrounded by marks "Trade_starts" and "Trade_ends"
            and include labels of confirming dialog slider, sales decisions ("Sell") and sharp price changes 
        prices at events ("current_date")
        prices indices corresponding to each event ("price_index") 
        values of slider ("slider_value", 0 for Sell, 3 for Keep)
        time stamps
        original logs of events as string values ("body")
"""
# reifev_df = pd.read_csv("/home/albelix/media/events_11_09_2021_21_07_19_export.csv",
#                         decimal='.', converters={'Index': str.strip})
reifev_df = pd.read_csv("/home/albelix/Documents/Reiffeisen/events_11_09_2021_21_07_19_export.csv",
                        decimal='.', converters={'Index': str.strip})
reifev_df['body'] = reifev_df['body'].apply(json.loads)
# events_10_27_2021_3.csv"), v.5: events_212.csv").
# use of fillna("") raises issues with strings, but may be needed, as otherwise creates ugly nan texts

# renaming relevant variables
reifev_df = reifev_df.rename(columns={reifev_df.columns[0]: 'participant.code'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[2]: 'round'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[9]: 'data'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[10]: 'Index'})

"""
6. Convering them to string types for merger
"""
reifev_df['participant.code'] = reifev_df['participant.code'].astype(str)
reifev_df['round'] = reifev_df['round'].astype(str)
reifev_df['data'] = reifev_df['data'].astype(str)
reifev_df['Index'] = reifev_df['Index'].astype(str)

# keep only one participant
reifev = reifev_df.loc[reifev_df['participant.code']==getpartcode].copy()

# checking data types
dataTypeSeries = reifev.dtypes
print(dataTypeSeries)
np.issubdtype(reifev['Index'].dtype, np.number) # checks if Index is pandas ds
#reifev['Index'] = reifev.loc['Index'].replace("", -1) # -1 for outcome selection stage

# (warnings here can be ignored. In fact, merger by Index are (most likely) not needed, as long as price values (data) are unique
# within participant and round. To be on the safe side, we also use Index as the last merger key)
reifev.to_csv('reifev.csv', index=False)  # save prepared events file


# Final preparations of subject data file:
# correct data type for one column:
dfsubj.iloc[:,8] = pd.to_numeric(dfsubj.iloc[:,8], errors='coerce') # puts nan if data is non-numeric.
# This is not problematic as long as such instances are not needed for further analysis

# Make sure values for merger in the main file are of same type
dfsubj['participant.code'] = dfsubj['participant.code'].astype(str)
dfsubj['round'] = dfsubj['round'].astype(str)
dfsubj['data'] = dfsubj['data'].astype(str)
dfsubj['Index'] = dfsubj['Index'].astype(str)

"""
7. merging decisions and events
"""
workdf = pd.merge(reifev, dfsubj,  how='outer', on=['participant.code', 'round', 'data', 'Index']).fillna(value=dfsubj, axis=1, inplace=False ) # bfill fills prev empty cells with next appropriate nonempty (for survey data)
print("merger of events")

# assigning proper types

workdf['round'] = workdf['round'].astype(int)
workdf['data'] = workdf['data'].astype(float) # this works once coerced  non-numeric chars.
workdf['data'] = pd.to_numeric(workdf['data']) #.astype(float) - best way to do
workdf['exit.price'] = pd.to_numeric(workdf['exit.price']) #.astype(float) - best way to do
workdf['Index'] = workdf['Index'].astype(float)
workdf['Index'] = workdf['Index'].astype('Int64')
# checking datatypes for comparison columns
print(workdf['data'].dtype)
print(workdf['exit.price'].dtype)

# dropping out Index elements not divisible by 10 (not operated anyway).
workdf=workdf[workdf['Index'] % 10 == 0]

# sorting data in proper order
workdf.sort_values(by=['participant.code', 'round', 'Index'], inplace=True)

# rearranging dataset for ease of readability
list(workdf.columns)
columnsTitles = ['participant.code', 'round',  'Index',  'data', 'exit.price', 'name',
                 'slider_value', 'clicked_volatility',  'volatility',  'owner__volatility', 'timestamp', 'unix_timestamp',
                 'secs_since_round_starts', 'body', 'obsno', 'session.code', 'buckser',  'survey.clear', 'survey.strategy', 'survey.dropdown', 'surv.debcard1', 'surv.credcard1', 'surv.conscred1', 'surv.mortgcred1', 'surv.bankacc1', 'surv.bankdepo1', 'surv.bankinv1', 'surv.invest1', 'surv.debcard2', 'surv.credcard2', 'surv.conscred2', 'surv.mortgcred2', 'surv.bankacc2', 'surv.bankdepo2', 'surv.bankinv2', 'surv.invest2', 'surv.invgoals', 'surv.invgoals_other', 'surv.inv_horiz', 'surv.norm7_know', 'surv.norm8_risk', 'surv.norm9_debt', 'surv.norm10_saving', 'surv.norm11_invlott', 'surv.norm11down_invlott', 'surv.norm11up_invlott', 'surv.riskat', 'surv.management', 'surv.age', 'surv.gender', 'surv.field', 'surv.field_other', 'surv.family', 'surv.famember', 'surv.income', 'surv.budget', 'surv.budget_other', 'surv.satis', 'surv.expect', 'surv.trust', 'surv.freedom', 'surv.group.id_in_subsession', 'surv.subsession.round_number', 'subject', 'TotalIndex']
workdf = workdf.reindex(columns=columnsTitles)


"""
8. Creation of decision variables
"""

# tagging sell point for each participant.code, range, exit.price==data
workdf['Diff']=np.where( (workdf['data'] == workdf['exit.price']) & (workdf['name']!=''),'1','0')
print(workdf['Diff'].describe())
print(workdf['Diff'].dtype)
print(workdf['Diff'].value_counts())

# delete all obs after sale
workdf['Diff'] = workdf['Diff'].astype(int)
workdf=workdf.groupby(['participant.code','round'],sort=False).apply(lambda x  : x.reset_index(
    drop=True).iloc[:x.reset_index(drop=True).Diff.idxmax()+1,:])
workdf.reset_index(drop=True, inplace=True) # needs to reset index

# checks # obs by groups and move variable to Choice
colist = workdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)
movd=workdf['Diff']
workdf.drop(labels=['Diff'], axis=1, inplace = True)
workdf.insert(5, 'Choice', movd)


# drop duplicates
workout = workdf.drop_duplicates(subset=['participant.code','round','Index'], keep='last')
g = workdf.groupby(['participant.code','round','Index'])
size = g.size()
var = size[size > 1] # autocorrect, was  just 'size[size>1]'
print("here")
print(var)


"""
9. Saving output file
"""

workout.to_csv('/home/albelix/media/workout.csv', mode="wt") # index=False, mode
print("mission accomplished")
