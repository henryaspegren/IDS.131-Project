# IDS.131-Project
KNOWN PROBLEMS:

1) Data has repeated entries

2) Data has birthdates that makes no sense

Data Cleaning / Filtering

1)Filtered data to contain first and second arrest only

2)Filtered data to contain second and third arrest only

3)[For recitivism prediction] Only selected users who are in the network (i.e they were arrested with other people in the first arrest)

FEATURES ADDITION:

1) age_cat -> quantile of their age in the dataset (1 to 10 categorial)

2)eigen - eigenvalue centrality in network

3)degree - degree centrality in network

4)closeness - closeness centrality in network

5)clus - local clustering coefficient in network

NETWORKS ARE BUILT BASED ON TIME(YEAR) OF ARREST SO WE DONT HAVE FUTURE INFORMATION

MODELS IN LOGSITIC REGRESSION / RANDOM FOREST:

1)BASELINE : ['first_arrest_SEXE','first_arrest_NCD1', 'first_arrest_MUN', 'first_arrest_ED1','age_cat']

2)SELF : BASELINE + ['eigen','degree','clus','closeness'] 

3)NEIGHBOUR : BASELINE + sum of neighbour's ['eigen','degree','clus','closeness'] 

4)SELF2010 : BASELINE + ['eigen','degree','clus','closeness']  from 2010 network (IE we use all the information to build the network , not just the information we have at the time of arrest - its cheating, just to see how well we can perform)



RESULTS FROM RUNNING K FOLDS ON RANNDOM FOREST CLASSIFIER.

****Fold=3*********

raw recidivism rate: 0.341884
1.0 percent of the people are arrested with someone and has networkinfo in this year[2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
data size is 88416
raw recidivism rate: 0.346736
wild guess accuracy is 0.653264115092

\==============baseline Data==========
Accuracy score of 0.879919923996 during training
ROC score of 0.95510035397 during training
Accuracy score of 0.881853963084 during training
ROC score of 0.955521046406 during training
Accuracy score of 0.881921824104 during training
ROC score of 0.955457901738 during training
The average validation score is 0.673430148389
The average validation AUC is 0.671329394353



\==============self Data==========
Accuracy score of 0.948798859935 during training
ROC score of 0.990768563388 during training
Accuracy score of 0.949799809989 during training
ROC score of 0.991362979168 during training
Accuracy score of 0.949087269273 during training
ROC score of 0.991049924299 during training
The average validation score is 0.711568041983
The average validation AUC is 0.733301674281



\==============self 2010 Data==========
Accuracy score of 0.95365092291 during training
ROC score of 0.989927188308 during training
Accuracy score of 0.953277687296 during training
ROC score of 0.990152837516 during training
Accuracy score of 0.952327633008 during training
ROC score of 0.989701611582 during training
The average validation score is 0.762746561708
The average validation AUC is 0.781303359362



\==============Neighbour Data==========
Accuracy score of 0.949104234528 during training
ROC score of 0.991029495065 during training
Accuracy score of 0.950071254072 during training
ROC score of 0.991137961165 during training
Accuracy score of 0.9493417481 during training
ROC score of 0.991170712162 during training
The average validation score is 0.698108939558
The average validation AUC is 0.71891503471


