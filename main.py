# Python script to extract subjects decisions data.

# The altorithm goes as follows:
# 1. Get data from original data file (data_2021-11-09_export.csv, for example)
# 2. Delete unneeded variables.
# 3. Get data for single subject, and
# 4. Reshape (expand) prices at the end of each batch of 10 to long format of 2500 obs. per participant.

# 5. Importing events data (events_11_09_2021_21_07_19_export.csv, for example)
# 6. Converting all mergCallinger variables to the same type and
# 7. Merging to the main workfile,
# 8. Defining conditions for dropdown and dropping data which are never observable because of that, and
# 9. Saving the working file as workout.csv

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi,  {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# 0. import modules
import pandas as pd
import json
import numpy as np
from ast import literal_eval
import itertools
import ast
import gc
import tracemalloc
import multiprocessing, random, sys, os, time

# make columns display wide for convenience
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# trace memory allocation
tracemalloc.start()

# 1. import the main data file, from app log (here, 'data_2021-11-09_export.csv', insert here correct name and address)
reifdat_df = pd.read_csv("/home/albelix/Documents/Reiffeisen/data_2021-11-09_export.csv") #data_2021-10_5.csv") #, converters={'backend.1.player.data':CustomParser},header=0)

# print column names and numbers
colist = reifdat_df.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)

# 2. drop inneeded columns by numbers
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

datdf=datdf.iloc[:,0:36]
colist = datdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)

print(datdf.head(10))
print("dataframe created")

# index for id value
print(type(datdf['participant.code']))
datdf['participant.code'] = datdf['participant.code'].astype(str)
datdf['subject'] = datdf['participant.code'].rank(method='first').astype(int)

# replacing unneeded prefixes in names
datdf.columns = datdf.columns.str.replace(".player", "")
datdf.columns = datdf.columns.str.replace("backend.", "")
datdf.columns = datdf.columns.str.replace("reif_survey_0.1", "surv")
for col in datdf.columns:
    print(col)
print(datdf.head(10))
print("all prepared")

# 3. explode price lists to rows for a single participant
colist = datdf.columns.to_list()
for (i, item) in enumerate(colist, start=0):
    print(i, item)
shapshot1 = tracemalloc.take_snapshot()
top_stats = shapshot1.statistics('lineno')

# 4. exploding (reshaping) cycle
dfin = np.empty((2500, 10), dtype=object)   # create empty df to collect reshaped data
d = datdf[datdf['subject'] == 1] # select data for single participant
for count in range(1,11,1):  # create main cycle over rounds
    colfull = [0, 1, 2, 3, 4, 4+(count-1)*3+1, 4+(count-1)*3+2, 4+(count-1)*3+3]   # pick up three variables in wide format
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
    dtm['data'] = dtm['data'].apply(ast.literal_eval)  # command needed to read json object as list
    dtnew = dtm.explode('data', ignore_index=False).reset_index()
    dtn = dtnew.values.tolist()
    # stack values for rounds 1:10
    if count==1:
        dfin[:250]=dtn
    else:
        dfin[250*(count-1):250*count]=dtn  #dsingle.append(dtn) this works but fills horizontally
        del [[dtm, dtnew]]  # clear temporary values
        gc.collect()
        dtm = pd.DataFrame()
        dtnew = pd.DataFrame()
        print()
    hdr.insert(0,'obsno')
    shapshot2 = tracemalloc.take_snapshot()
    top_stats = shapshot2.statistics('lineno')
    # speed of processing report
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

print("reach")
dfsubj = pd.DataFrame(dfin, columns=hdr)
print(type(dfsubj))
# creating indices
dfsubj['TotalIndex'] = np.arange(len(dfsubj)) # total index of observation per participant, python format
dfsubj["Index"] = dfsubj.groupby("round")["TotalIndex"].rank(method="first", ascending=True) # observation index per round
# savind dataframe
dfsubj.to_csv('data_subject.csv', index=False)
getpartcode=dfsubj['participant.code'][0] # get code of the current participant
print("decisions database completed")

# Commented code below is for reference purposes
# # code to explode data for several participants incycle
# #dfin=[]
# dfin = np.empty((130000,62), dtype=object)   # insert here 2500*# of subjects, 10
# #dsingle=[]
# dsingle = np.empty((2500, 62), dtype=object)  # for dataset 3, use (2500,60)
# d = {}
# print("2getdata")
# iter=0
# print(len(datdf['subject'])+1)
# for subj in range(1,len(datdf['subject'])+1):
#     d[subj] = datdf[datdf['subject'] == subj]
#     print(d[subj].iloc[:, 0:8].head(10))
#     for count in range(1,11,1):
#     #    print(count)
#         cols = [0,1,2,3,4,5,5+(count-1)*3+1, 5+(count-1)*3+2, 5+(count-1)*3+3]
#         My_list = list(range(36, 87)) # for dataset 3, use (36, 85)
#         colfull = cols+My_list
#     #   print(colfull)
#         # dtmp=d[count]
#         dtm = d[subj].iloc[:,colfull] #select colums by numbers dtmp.filter(colfull) # dtmp[np.intersect1d(dtmp.columns, colfull)]
#         dtm = dtm.rename(columns={dtm.columns[4]: 'compnumber'})
#         dtm = dtm.rename(columns={dtm.columns[5]: 'buckser'})
#         dtm= dtm.rename(columns={dtm.columns[6]: 'volatility'})
#         dtm= dtm.rename(columns={dtm.columns[7]: 'data'})
#         dtm= dtm.rename(columns={dtm.columns[8]: 'exit.price'})
#     #    print(dtm.iloc[:,0:9].head(10))
#         #dtm['data'] = dtm['data'].apply(ast.literal_eval)  # command needed to read json object as list
#         #dtm['data'] = dtm['data'].str.strip('()').str.split(',')
#         #dtm = pd.concat([dtm, dtm['data'].apply(pd.Series)], axis=1)
#         dtm['round'] = count
#     #    print(dtm['round'])
#         hdr = dtm.columns.values.tolist()
#     #    print(dtm['data'])
#     #    print("dtm data type is", type(dtm['data']))
#         #dtm['data'] = dtm['data'].apply(literal_eval)  # command needed to read json object as list
#     #    print("mission started")
#         if count==1:
#             dtm['data'] = dtm['data'].str.split(',')
#             dtnew = dtm.explode('data', ignore_index=False).reset_index()
#     #        print(dtnew.iloc[:,0:9].head(10))
#             dtn = dtnew.values.tolist()
#     #        print("dtn data type is", type(dtn))
#         else:
#             dtm['data'] = dtm['data'].str.split(',')
#             #dtm['data'] = dtm['data'].str.strip('[]').astype(float)
#             dtnew = dtm.explode('data', ignore_index=False).reset_index()
#     #        print(dtnew.iloc[:,0:9].head(10))
#             dtn = dtnew.values.tolist()
#     #        print("dtn data type is", type(dtn))
#     #    print("mission accomplished")
#         if count==1:
#             dsingle[:250]=dtn
#         else:
#             dsingle[250*(count-1):250*count]=dtn  #dsingle.append(dtn) this works but fills horizontally
#     #        x, y = dsingle[-15:-1, 0:9], dsingle[-15:-1, 55:57]
#     #        print(x,y)
#     #        print("iteration")
#             # dtm = None
#             # dtnew = None
#             del [[dtm, dtnew]]
#             gc.collect()
#             dtm = pd.DataFrame()
#             dtnew = pd.DataFrame()
#             print()
#     if subj==1:
#         dfin[:2500]=dsingle
#     else:
#         dfin[2500*(subj-1):2500*subj]=dsingle  # np.vstack([dfin, dsingle])
#         hdr.insert(0,'obsno')
#         shapshot2 = tracemalloc.take_snapshot()
#         top_stats = shapshot2.statistics('lineno')
#
#         print("[ Top 10 ]")
#         for stat in top_stats[:10]:
#             print(stat)
#
# print("reach")
# dfsubj = pd.DataFrame(dfin, columns=hdr)
# # dfsubj.replace({'data': {'[': '', ']': ''}}, regex=True)
# dfsubj.to_csv('data_stud52.csv', index=False) # old - data_33.csv
# # instead of next line, I just drop extra [ ]  in excel
# #dfsubj['data'] = dfsubj['data'].str.replace(r'[][]', '', regex=True) # to remove reminders of [ ]
# print("decisions database completed")

# block to check extra memory use
# if __name__ == '__main__':
#     manager = multiprocessing.Manager()
#     state = manager.dict(list_size=5*1000*1000)  # shared state
#     p = multiprocessing.Process(target=run_test, args=(state,))
#     p.start()
#     p.join()
#     print('time to sort: %.3f' % state['time'])
#     print('my PID is %d, sleeping for a minute...' % os.getpid())
#     time.sleep(60)
    # at this point you can inspect the running process to see that it
    # does not consume excess memory


# 5. Calling of event data
reifev_df = pd.read_csv("/home/albelix/Documents/Reiffeisen/events_11_09_2021_21_07_19_export.csv").fillna("") # version 3: events_10_27_2021_3.csv"), v.5: events_212.csv"). fillna is needed, as otherwise creates ugly nan texts
reifev_df['body'] = reifev_df['body'].apply(json.loads)

# renaming relevan variables
reifev_df = reifev_df.rename(columns={reifev_df.columns[0]: 'participant.code'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[2]: 'round'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[9]: 'data'})
reifev_df = reifev_df.rename(columns={reifev_df.columns[10]: 'Index'})

# 6. Convering them to proper types for merger
reifev_df['participant.code'] = reifev_df['participant.code'].astype(str)
reifev_df['round'] = reifev_df['round'].astype(str)
reifev_df['data'] = reifev_df['data'].astype(str)

# keep only one participant
reifev = reifev_df.loc[reifev_df['participant.code']==getpartcode]

# converting Index of obs within round to integer for merger
reifev['Index'] = reifev['Index'].replace('', -1) # -1 for outcome selection stage
reifev['Index'] = reifev['Index'].astype(int)
# (warnings here can be ignored. In fact, merger by Index are (most likely) not needed, as long as price values (data) are unique
# within participant and round. To be on the safe side, we also use Index as the last merger key)

# make sure values for merger in the main file are of same type
dfsubj['participant.code'] = dfsubj['participant.code'].astype(str)
dfsubj['round'] = dfsubj['round'].astype(str)
dfsubj['data'] = dfsubj['data'].astype(str)
dfsubj['Index'] = dfsubj['Index'].astype(int)

# 7. merging decisions and events
workdf = pd.merge(reifev, dfsubj,  how='outer', on=['participant.code', 'round', 'data', 'Index']).fillna(value=dfsubj, axis=1, inplace=False ) # bfill fills prev empty cells with next appropriate nonempty (for survey data)
print("merger of events")

workdf=workdf[workdf['Index'] % 10 == 0]

workdf['round'] = workdf['round'].astype(int)
workdf['data'] = workdf['data'].astype(float)
workdf['Index'] = workdf['Index'].astype(int)
workdf.sort_values(by=['round', 'Index'], inplace=True)

# 8. Creation of decision variables
workdf["choice"]=0
#workout=workdf.copy()
# define end of round actions
for i in range(len(workdf)): # range is 0:N-1
    workdf['choice'] = (((workdf["name"] == "slider value changed") & (workdf['slider_value'] == 0)) | (workdf['exit.price'] == workdf['data'])).astype(int)

# define rows after end of the round to drop
ix = (workdf['choice'].eq(1)
      .cumsum()
      .groupby(workdf['round'])
      .apply(lambda x: x.loc[x.idxmax():]).index.get_level_values(1))
workout=workdf.drop(ix,axis=0) # keeping only relevant rows

# dropping extra observations
# workout.drop_duplicates(subset = ("data"), keep = False, inplace = True)

# 9. Saving output
workout.to_csv('/home/albelix/Documents/Reiffeisen/workout.csv', index=False)
print("mission accomplished")
