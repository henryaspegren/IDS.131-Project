#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:10:21 2017

@author: kai
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import copy

from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.ensemble
import sklearn as skl
#%%
def getCooffending():
    co_offending_table = pd.read_csv('./Cooffending.csv')
    
    # Remove duplicate rows
    co_offending_table.drop_duplicates(inplace=True)
    
    # Format the date column as a python datetime object
#==============================================================================
#     co_offending_table['Date'] = co_offending_table.Date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
#==============================================================================
    co_offending_table['Date']=pd.to_datetime(co_offending_table.Date)
    co_offending_table['year']=co_offending_table.Date.dt.year
    co_offending_table['month']=co_offending_table.Date.dt.month
    # Add a column for the number of arrests of each offender
#==============================================================================
#     co_offending_table['ArrestCount'] = co_offending_table.groupby('NoUnique')['NoUnique'].transform('count')
#     
#     # Get the right datatype 
#     co_offending_table.SeqE = co_offending_table.SeqE.astype('category')
#     co_offending_table.SEXE = co_offending_table.SEXE.astype('category')
#     co_offending_table.NCD1 = co_offending_table.NCD1.astype('category')
#     co_offending_table.NCD2 = co_offending_table.NCD2.astype('category')
#     co_offending_table.NCD3 = co_offending_table.NCD3.astype('category')
#     co_offending_table.NCD4 = co_offending_table.NCD4.astype('category')
#     co_offending_table.MUN = co_offending_table.MUN.astype('category')
#     co_offending_table.ED1 = co_offending_table.ED1.astype('category')
#==============================================================================
    return co_offending_table



def readTrainingData(filename):
    training_data = pd.read_csv(filename)
    
    def addMajorCrimeCategory(train):
        mapper=pd.read_csv('crimeType.csv')
        crimeTypeDict={' ':'x'}
        def getDict(row):
            crimeTypeDict[str(row.NCD)]=row.Type
        mapper.apply(getDict,1)
        crimeTypeDict.keys()
        train['first_arrest_crimeType']=train.first_arrest_NCD1.apply(lambda x : crimeTypeDict[x]).astype('category')
    
    addMajorCrimeCategory(training_data)
    training_data['age_cat']=pd.qcut(training_data.first_arrest_Naissance,10,labels=False).astype('category')
    training_data['arrestedWithAdult']=(training_data['first_arrest_Adultes']>=2).astype(int)
    training_data['arrestedWithYouth']=np.sign(training_data['first_arrest_Jeunes'])
    training_data['arrestedWithSomeone']=np.sign(training_data['first_arrest_Adultes']+training_data['first_arrest_Jeunes'])
    training_data['first_arrest_Date']=pd.to_datetime(training_data.first_arrest_Date)
    training_data['first_arrest_year']=training_data.first_arrest_Date.dt.year
    # format data
    training_data.first_arrest_SeqE = training_data.first_arrest_SeqE.astype('category')
    training_data.first_arrest_SEXE = training_data.first_arrest_SEXE.astype('category')
    training_data.first_arrest_NCD1 = training_data.first_arrest_NCD1.astype('category')
    training_data.first_arrest_NCD2 = training_data.first_arrest_NCD2.astype('category')
    training_data.first_arrest_NCD3 = training_data.first_arrest_NCD3.astype('category')
    training_data.first_arrest_NCD4 = training_data.first_arrest_NCD4.astype('category')
    training_data.first_arrest_MUN = training_data.first_arrest_MUN.astype('category')
    training_data.first_arrest_ED1 = training_data.first_arrest_ED1.astype('category')
    training_data.second_arrest_SeqE = training_data.second_arrest_SeqE.astype('category')
    training_data.second_arrest_SEXE = training_data.second_arrest_SEXE.astype('category')
    training_data.second_arrest_NCD1 = training_data.second_arrest_NCD1.astype('category')
    training_data.second_arrest_NCD2 = training_data.second_arrest_NCD2.astype('category')
    training_data.second_arrest_NCD3 = training_data.second_arrest_NCD3.astype('category')
    training_data.second_arrest_NCD4 = training_data.second_arrest_NCD4.astype('category')
    training_data.second_arrest_MUN = training_data.second_arrest_MUN.astype('category')
    training_data.second_arrest_ED1 = training_data.second_arrest_ED1.astype('category')
    return training_data



#%%
def getNetwork(df,year):
    crimes=df.groupby('SeqE')
    arrestedPerCrime=crimes.apply(len)
    multiPersonCrime=arrestedPerCrime[arrestedPerCrime!=1].index
    multiPersonCrime=df[df.SeqE.isin(multiPersonCrime)]
    nodes=multiPersonCrime.NoUnique.unique()
    print 'network has {} nodes'.format(len(nodes))
    G=nx.Graph()
    G.add_nodes_from(nodes)
    multCrimes=multiPersonCrime.groupby('SeqE')
    for seqe,crime in multCrimes:
        people=list(crime.NoUnique)
        for i in range(len(people)):
            for j in range(i+1,len(people)):
                G.add_edge(people[i],people[j])
                
    print 'network has {} edges'.format(len(G.edges()))
    nx.write_gpickle(G,'G_crime_{}.pkl'.format(year))

#==============================================================================
# def getNetwork2(df,year):
#     multiPersonCrime=df[df.Adultes>=2]
#     nodes=multiPersonCrime.NoUnique.unique()
#     print 'network has {} nodes'.format(len(nodes))
#     G=nx.Graph()
#     G.add_nodes_from(nodes)
#     multCrimes=multiPersonCrime.groupby('SeqE')
#     for seqe,crime in multCrimes:
#         people=list(crime.NoUnique)
#         for i in range(len(people)):
#             for j in range(i+1,len(people)):
#                 G.add_edge(people[i],people[j])
#                 
#     print 'network has {} edges'.format(len(G.edges()))
#     nx.write_gpickle(G,'2G_crime_{}.pkl'.format(year))
#==============================================================================
    
def getNetworks(df):
    years=list(df.year.unique())
    for year in years:
        dfyear=df[df.year<=year]
        getNetwork(dfyear,year)
        

    
def computeNetworkValues(G):
    eigen=nx.eigenvector_centrality(G)
    degree=G.degree()
    closeness=nx.closeness_centrality(G)
#==============================================================================
#     between=nx.betweenness_centrality(G)
#==============================================================================
    clus=nx.clustering(G)
    return degree,eigen,clus,closeness

def computeNetworksValues(df):
#==============================================================================
#     years=list(df.year.unique())
#==============================================================================
    networkFeatures={}
    for year in [2010]:
        networkFeatures[year]={}
        G=nx.read_gpickle('2G_crime_{}.pkl'.format(year))
        networkFeatures[year]['degree'],networkFeatures[year]['eigen'],networkFeatures[year]['clus'],networkFeatures[year]['closeness']=computeNetworkValues(G)
    json.dump(networkFeatures, open('networkFeatures2010.json', 'wb'))
def runForNetworkVal(read=True):
    co_offending_table = pd.read_csv('./Cooffending.csv')
    
    # Remove duplicate rows
    co_offending_table.drop_duplicates(inplace=True)
    
    # Format the date column as a python datetime object
    co_offending_table['Date'] = co_offending_table.Date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    co_offending_table['year']=co_offending_table.Date.dt.year
    co_offending_table['month']=co_offending_table.Date.dt.month
    # Add a column for the number of arrests of each offender
    co_offending_table['ArrestCount'] = co_offending_table.groupby('NoUnique')['NoUnique'].transform('count')
    
    # Get the right datatype 
    co_offending_table.SeqE = co_offending_table.SeqE.astype('category')
    co_offending_table.SEXE = co_offending_table.SEXE.astype('category')
    co_offending_table.NCD1 = co_offending_table.NCD1.astype('category')
    co_offending_table.NCD2 = co_offending_table.NCD2.astype('category')
    co_offending_table.NCD3 = co_offending_table.NCD3.astype('category')
    co_offending_table.NCD4 = co_offending_table.NCD4.astype('category')
    co_offending_table.MUN = co_offending_table.MUN.astype('category')
    co_offending_table.ED1 = co_offending_table.ED1.astype('category')
    if not read:
        getNetworks(co_offending_table)
    computeNetworksValues(co_offending_table)

def getCrimeDict(co_offending_table):
    crimeDict={}
    crimes=co_offending_table.groupby('SeqE')
    def getOffenders(chunk):
        crimeID=chunk.SeqE.unique()[0]
        crimeDict[crimeID]=list(chunk.NoUnique)
    crimes.apply(getOffenders)
    json.dump(crimeDict, open('crimeDict.json', 'wb'))
    
#==============================================================================
# runForNetworkVal(read=False)
#==============================================================================
#==============================================================================
# df=getCooffending()
# computeNetworksValues(df)
#==============================================================================

#%%
def getNetworkVar(row,networkFeatures,crimeDict,Gs):
    year=row.first_arrest_year
    G=Gs[year]
    year=str(year)
    eigen,degree,clus,closeness=0,0,0,0
    crime=str(row.first_arrest_SeqE)
    people=copy.deepcopy(crimeDict[crime])
    offender=row.first_arrest_NoUnique
    people.remove(offender)
    networkFeat=networkFeatures[year]
    for person in people:
        try:
            sperson=str(person)
            eigen+=networkFeat['eigen'][sperson]
            degree+=networkFeat['degree'][sperson]
            clus+=networkFeat['clus'][sperson]
            closeness+=networkFeat['closeness'][sperson]
        except:
            pass
    count=0
    for i in range(len(people)):
        for j in range(i+1,len(people)):
            count+=int(G.has_edge(people[i],people[j]))
    total=len(people)*(len(people)-1)/2
    if total!=0:
        clus2=count*1.0/total
    else:
        clus2=0
    return eigen,degree,clus,clus2,closeness

def addNetworkFeatures(co_offending_table,df):
    years=list(co_offending_table.year.unique())
    Gs={}
    for year in years:
        Gs[year]=nx.read_gpickle('G_crime_{}.pkl'.format(year))
    networkFeatures=json.load(open('networkFeatures.json','rb'))
    crimeDict=json.load(open('crimeDict.json','rb'))
    df2=pd.DataFrame(df.first_arrest_NoUnique)
    df2['Neigen'],df2['Ndegree'],df2['Nclus'],df2['Nclus2'],df2['Ncloseness']=zip(*df[['first_arrest_year','first_arrest_NoUnique','first_arrest_SeqE']].apply(lambda x: getNetworkVar(x,networkFeatures,crimeDict,Gs),axis=1))
    df2.to_csv('networkFeatures1_23.csv',index=False)
    pass



def getNetworkVar2(row,networkFeatures):
    year=row.first_arrest_year
    year=str(year)
    offender=row.first_arrest_NoUnique
    networkFeat=networkFeatures[year]
    sperson=str(offender)
    try:
        eigen=networkFeat['eigen'][sperson]
        degree=networkFeat['degree'][sperson]
        clus=networkFeat['clus'][sperson]
        closeness=networkFeat['closeness'][sperson]
    except:
        eigen=0
        degree=0
        clus=0
        closeness=0
        print (sperson)
    return eigen,degree,clus,closeness

def addNetworkFeatures2(co_offending_table,trainsub):
    networkFeatures=json.load(open('networkFeatures.json','rb'))
    df=pd.DataFrame(trainsub.first_arrest_NoUnique)
    df['eigen'],df['degree'],df['clus'],df['closeness']=zip(*trainsub[['first_arrest_year','first_arrest_NoUnique','first_arrest_SeqE']].apply(lambda x: getNetworkVar2(x,networkFeatures),axis=1))
    df.to_csv('networkFeatures2_23.csv',index=False)

def getNetworkVar3(row,networkFeatures):
    year=row.first_arrest_year
    year=str(2010)
    offender=row.first_arrest_NoUnique
    networkFeat=networkFeatures[year]
    sperson=str(offender)
    try:
        eigen=networkFeat['eigen'][sperson]
        degree=networkFeat['degree'][sperson]
        clus=networkFeat['clus'][sperson]
        closeness=networkFeat['closeness'][sperson]
    except:
        eigen=0
        degree=0
        clus=0
        closeness=0
        print (sperson)
    return eigen,degree,clus,closeness

def addNetworkFeatures3(co_offending_table,trainsub):
    networkFeatures=json.load(open('networkFeatures2010.json','rb'))
    df=pd.DataFrame(trainsub.first_arrest_NoUnique)
    df['eigen2010'],df['degree2010'],df['clus2010'],df['closeness2010']=zip(*trainsub[['first_arrest_year','first_arrest_NoUnique','first_arrest_SeqE']].apply(lambda x: getNetworkVar3(x,networkFeatures),axis=1))
    df.to_csv('networkFeatures3_23.csv',index=False)
    

#==============================================================================
# co_offending_table=getCooffending()
#==============================================================================
#==============================================================================
# computeNetworksValues(co_offending_table)
#==============================================================================
#==============================================================================
# getNetworks(co_offending_table)
#==============================================================================
#==============================================================================
# training_data=readTrainingData('secondthirdarrest.csv')
# training_subset=training_data[(training_data.arrestedWithAdult==1)]
# addNetworkFeatures(co_offending_table,training_subset.iloc[:,:])
# addNetworkFeatures2(co_offending_table,training_subset.iloc[:,:])
# addNetworkFeatures3(co_offending_table,training_subset.iloc[:,:])
#==============================================================================
#%%
#==============================================================================
# G=nx.read_gpickle('2G_crime_2010.pkl')
# G_s=nx.read_gpickle('G_crime_2010.pkl')
#==============================================================================


#%%

def runLogPrediction(training_data,testing_data,xVars):
    X_df = training_data[xVars]
    # gives us dummy variables
    X_df = pd.get_dummies(X_df)
    X = X_df.as_matrix()

    Y_df = training_data[['arrested_again']]
    Y = Y_df.as_matrix()
    Y = Y.ravel()
    baseline_model = LogisticRegression(penalty='l2')
    baseline_model.fit(X, Y)
    
    res = np.argsort(abs(baseline_model.coef_))[0]
    res = res[::-1]
    print('bias: %f' % baseline_model.intercept_)
    for coeff_index in res[0:20]:
        value = baseline_model.coef_[0][coeff_index]
        name = X_df.columns[coeff_index]
        print('coefficient: %s  | value: %f' % (name, value))
    print('Accuracy score of {} during training'.format(baseline_model.score(X, Y)))
    Yprob=baseline_model.predict_proba(X)[:,1]
    roc=roc_auc_score(Y, Yprob)
    print ('ROC score of {} during training'.format(roc))
    X_test_df = testing_data[xVars]
    # gives us dummy variables
    X_test_df = pd.get_dummies(X_test_df)
    X_test = X_test_df.as_matrix()
    Y_test_df = testing_data[['arrested_again']]
    Y_test= Y_test_df.as_matrix()
    Y_test = Y_test.ravel()
    testscore=baseline_model.score(X_test, Y_test)
    Ytestprob=baseline_model.predict_proba(X_test)[:,1]
    testroc=roc_auc_score(Y_test, Ytestprob)
    return testscore,testroc

def runRandomForest(training_data,testing_data,xVars,n,k):
    X_df = training_data[xVars]
    # gives us dummy variables
    X_df = pd.get_dummies(X_df)
    X = X_df.as_matrix()

    Y_df = training_data[['arrested_again']]
    Y = Y_df.as_matrix()
    Y = Y.ravel()
    
    classifer=skl.ensemble.RandomForestClassifier(n_estimators=n,max_features=k,bootstrap=True,oob_score=False)
    classifer.fit(X,Y)
    
    print('Accuracy score of {} during training'.format(classifer.score(X, Y)))
    Yprob=classifer.predict_proba(X)[:,1]
    roc=roc_auc_score(Y, Yprob)
    print ('ROC score of {} during training'.format(roc))
    X_test_df = testing_data[xVars]
    # gives us dummy variables
    X_test_df = pd.get_dummies(X_test_df)
    X_test = X_test_df.as_matrix()
    Y_test_df = testing_data[['arrested_again']]
    Y_test= Y_test_df.as_matrix()
    Y_test = Y_test.ravel()
    testscore=classifer.score(X_test, Y_test)
    Ytestprob=classifer.predict_proba(X_test)[:,1]
    testroc=roc_auc_score(Y_test, Ytestprob)
    return testscore,testroc

def kFOlD(k,trainingdata,predictor,xVars):
    df = shuffle(trainingdata)
    def partition(df, n): 
        division = len(df) / float(n) 
        return [ df.iloc[int(round(division * i)): int(round(division * (i + 1))),:] for i in xrange(n) ]
    dfs=partition(df,k)
    testscores=[]
    aucscores=[]
    for i in range(k):
        training=copy.deepcopy(dfs)
        del training[i]
        training=pd.concat(training)
        validation=dfs[i]
        testscore,aucscore=predictor(training,validation,xVars)
        testscores.append(testscore)
        aucscores.append(aucscore)
    print('The average validation score is {}'.format(np.mean(testscores)))
    print('The average validation AUC is {}'.format(np.mean(aucscores)))
    return testscores,aucscores

#training_data = build_table_of_first_two_arrests(co_offending_table)
# or read from csv
training_data=readTrainingData('./basic_model_data.csv')
print('raw recidivism rate: %f' % (1.0*sum(training_data.arrested_again)/len(training_data.arrested_again)))

#first-second arrest
training_subset=training_data[(training_data.arrestedWithAdult==1)]
networkData1=pd.read_csv('networkFeatures1.csv')
networkData2=pd.read_csv('networkFeatures2.csv')
networkData3=pd.read_csv('networkFeatures3.csv')
training_subset=pd.merge(training_subset,networkData1,how='left',on='first_arrest_NoUnique')
training_subset=pd.merge(training_subset,networkData2,how='left',on='first_arrest_NoUnique')
training_subset=pd.merge(training_subset,networkData3,how='left',on='first_arrest_NoUnique')
year=[2003]
year=[2003,2004,2005,2006,2007,2008,2009,2010]
training_yearsub=training_subset[training_subset.first_arrest_year.isin(year)]
siz=1.0*len(training_yearsub)/len(training_subset)
print('{} percent of the people are arrested with someone and has networkinfo in this year{}'.format(siz,year))
print ('data size is {}'.format(len(training_yearsub)))
print('raw recidivism rate: %f' % (1.0*sum(training_yearsub.arrested_again)/len(training_yearsub.arrested_again)))
print ('wild guess accuracy is {}'.format(1-1.0*sum(training_yearsub.arrested_again)/len(training_yearsub.arrested_again)))


print('\n\n\n\==============baseline Data==========')
fold=10
n=100
k=10
xVars=['first_arrest_SEXE','first_arrest_NCD1', 'first_arrest_MUN', 'first_arrest_ED1','age_cat']
testscores,aucscores=kFOlD(fold,training_yearsub,lambda x,y,z: runRandomForest(x,y,z,n,k),xVars)
json.dump(testscores,open('{}testscore_base.json'.format(k),'wb'))
json.dump(aucscores,open('{}aucscore_base.json'.format(k),'wb'))


print('\n\n\n\==============self Data==========')
fold=10
n=100
k=10
xVars=['first_arrest_SEXE','first_arrest_NCD1', 'first_arrest_MUN', 'first_arrest_ED1','age_cat']+['eigen','degree','clus','closeness'] 
testscores,aucscores=kFOlD(fold,training_yearsub,lambda x,y,z: runRandomForest(x,y,z,n,k),xVars)
json.dump(testscores,open('{}testscore.json'.format(k),'wb'))
json.dump(aucscores,open('{}aucscore.json'.format(k),'wb'))


print('\n\n\n\==============self 2010 Data==========')
fold=10
n=100
k=10
xVars=['first_arrest_SEXE','first_arrest_NCD1', 'first_arrest_MUN', 'first_arrest_ED1','age_cat']+['eigen2010','degree2010','clus2010','closeness2010'] 
testscores,aucscores=kFOlD(fold,training_yearsub,lambda x,y,z: runRandomForest(x,y,z,n,k),xVars)
json.dump(testscores,open('{}testscore_2010.json'.format(k),'wb'))
json.dump(aucscores,open('{}aucscore_2010.json'.format(k),'wb'))



print('\n\n\n\==============Neighbour Data==========')
fold=10
n=100
k=10
xVars=['first_arrest_SEXE','first_arrest_NCD1', 'first_arrest_MUN', 'first_arrest_ED1','age_cat']+['Neigen','Ndegree','Nclus','Nclus2','Ncloseness'] 
testscores,aucscores=kFOlD(fold,training_yearsub,lambda x,y,z: runRandomForest(x,y,z,n,k),xVars)
json.dump(testscores,open('{}testscore_neigh.json'.format(k),'wb'))
json.dump(aucscores,open('{}aucscore_neigh.json'.format(k),'wb'))