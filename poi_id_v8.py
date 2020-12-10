
# coding: utf-8

# In[1]:


### Setting up the environment
import os
os.getcwd()
import sys
sys.path.append("..\\tools")
### print (sys.path)


# In[2]:


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
import pickle

### Feature selection - I'm attempting to limit features based on
### their description.
### I've omitted features related to deferral payments
### Don't think those are relevant for identifying POI
### Have included all other features below
features_list = ['poi','salary','total_payments','loan_advances', 'bonus','restricted_stock_deferred', 
'deferred_income','total_stock_value', 'expenses', 'exercised_stock_options','other', 'long_term_incentive', 
'restricted_stock','to_messages', 'from_poi_to_this_person','from_messages', 
'from_this_person_to_poi','shared_receipt_with_poi']


# In[3]:


### Am loading data and converting it into a dataframe with each feature in
### a separate column. This will make it convenient to plot and find 
### outliers

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
my_dataset = data_dict
print "input dataset", my_dataset.items()[0:2]

# In[4]:


import matplotlib.pyplot as plt
import pylab
### get_ipython().magic(u'pylab inline')

### For feature selection,
### Lets check on a count of NaN values for each feature. 
### We will use this to decide on the features to use for evaluation

def count_nan (the_dataset):
    """ This function returns a count of NaN entries per feature"""
    NaNdata = featureFormat(my_dataset, features_list,sort_keys = False, 
    remove_NaN = False,remove_all_zeroes=False)
    nancount = {}
    nanFrame = pd.DataFrame(data=NaNdata, columns=features_list)
    for columnName in nanFrame:
###        print nanFrame[columnName]
        counts = sum(nanFrame[columnName].isin(['NaN']).values)
###       print "column name", columnName, "value is", nanFrame[columnName]
###       print "counts for ", columnName, "is ", counts
        nancount[columnName] = counts
###        print "nancount", columnName, counts
    return nancount

### Get the count of NaNs for each feature and display them in a graph
countDict = count_nan(my_dataset)
### print "Countdict", countDict
##print "values", countDict.values()
xcoords = np.arange(len(countDict.keys()))
##print "xcoords is", xcoords
fig, baraxes = plt.subplots()
fig.set_size_inches(20,6)
plt.xlabel('Feature Name')
plt.ylabel('Count of NaNs')
plt.xticks(rotation=30)
plt.bar(xcoords, countDict.values())
xc, xp = plt.xticks(xcoords, countDict.keys())


# ## Feature selection
# 
# From the graph, the feature 'Loan advances' has the maximum NaN entries. This is followed by 'Restricted stock deferred', 'deferred income' and 'long_term_incentive'. These four features have over half the entries as 'NaN',i.e. over 70 of the 146 entries are NaN. 
# 
# These features will be ignored. The other features have less than 60 entries as NaN, which is as good a coverage as we can get for most features. We will work with that.
# 
# I'll also remove some of the other features that may not be relevant. This includes 'restricted stock', 'Expenses' and 'other'
# 

# In[5]:


### removing the following columns from the dataframe
### 'restricted_stock_deferred", 'deferred_income', 'loan_advances' 

features_list.remove('loan_advances')
features_list.remove('restricted_stock_deferred')
features_list.remove('deferred_income')
features_list.remove('long_term_incentive')
### features_list.remove('expenses')
### features_list.remove('restricted_stock')
### features_list.remove('other')
wNaNdata = featureFormat(my_dataset, features_list,sort_keys = False, remove_NaN = False,                             remove_all_zeroes=False)
wNaNFrame = pd.DataFrame(data=wNaNdata, columns=features_list)
print "features list", features_list 

### Lets try to get a count of POIs and non POIs in the dataset
countPOIs = len(wNaNFrame[wNaNFrame['poi'] == 1.0])
countNoPOIs = len(wNaNFrame[wNaNFrame['poi'] == 0.0])

print "POI count ", countPOIs
print "non POI count", countNoPOIs

# ## Graphs to identify outliers
# 
# Lets try to plot each of these features to identify any outliers.

# In[6]:


### I'll plot data points to identify outliers

figurenum = 1
for column in wNaNFrame:
    plt.figure(figurenum, figsize=(20,6))
    plt.scatter(wNaNFrame.index, wNaNFrame[column],label = column)
    pylab.legend(loc='upper left')
    figurenum += 1


# ## Identify and remove Outliers
# The figures for the financial features show atleast one outlier. 
# It seems to have an index of slightly above 100, and seems to be 
# the same row for all financial features. 
# In this section we identify and remove it.

# In[7]:


### I'll only plot the indexes from 100-110 to confirm and identify the outlier

figurenum = 1
for column in wNaNFrame:
    plt.figure(figurenum, figsize=(20,6))
    plt.scatter(wNaNFrame.index[100:110], wNaNFrame[column][100:110],label = column)
    pylab.legend(loc='upper left')
    figurenum += 1


# In[8]:


## Identify and remove outlier

##The max value seems to be 104, and is not a manually identified poi
##Lets check the index associated name

max_name = my_dataset.keys()[104]
print "Name of employee with max values is", max_name

wNaNFrame[104:105]


# In[9]:


### Lets remove that entry from the dictionary, 
### since it is a fabricated/calculated entry
### and does not represent a real person

del my_dataset['TOTAL']
wNaNFrame.drop(wNaNFrame.index[104], inplace=True)
print wNaNFrame[104:105]


## Lets run the graphs again to confirm and identify
## any additional outliers

figurenum = 1
for column in wNaNFrame:
    plt.figure(figurenum, figsize=(20,6))
    plt.scatter(wNaNFrame.index, wNaNFrame[column],label = column)
    pylab.legend(loc='upper left')
    figurenum += 1


# # Prepare rows with NaN
# The dataframe has cells with a "NaN" string.
# I'll replace it with NaN value, so it is easier 
# to remove rows with NaN later.

# In[10]:


### Drop rows with NA and check on number of rows available
###wFrame = wNaNFrame.dropna(axis=0,how='any', inplace=True)

## Replacing "NaN" string values to NaN
wNaNFrame.replace(to_replace="NaN", value=np.NaN, inplace=True)
        


# # Split dataframe into financial and email data
# 
# It seems the financial data and email data has NaN values for different 
# indices. If we try to remove rows with NaN across the entire dataset,
# we will be left with few rows for analysis.
# 
# I expect by splitting the dataframe into two, we will be able to get more rows 
# that are populated completely. This will mean we have to evaluate the 
# financial data separately from the email data.

# In[11]:


### Separating the data into two separate frames
### One for financial, and another for email.

wNaNFinFrame = wNaNFrame.iloc[:,:9]
wNaNEmailFrame = wNaNFrame.iloc[:, 9:]
wNaNEmailFrame.loc[:,'poi'] = wNaNFrame.loc[ :,'poi']



# # Remove NaN entries 
# 
# I'll remove NaN entries from the financial
# and email dataframes to check on number of entries
# with data

# In[12]:


### use dropna to remove all entries with NaN

wFinFrame = wNaNFinFrame.dropna(axis='index', how='any')
wFinFrame.reset_index(drop=True, inplace=True)
wEmailFrame = wNaNEmailFrame.dropna(axis='index', how ='any')
wEmailFrame.reset_index(drop=True, inplace=True)

print "Count of financial rows", wFinFrame.shape[0]
print "Count of Email rows", wEmailFrame.shape[0]
### print "Top financial rows", wFinFrame.head()
### print "Top email rows", wEmailFrame.head()



# # Using 2 separate datasets
# Since I have more complete data rows available
# with 2 separate datasets, I'll continue with 
# further investigations using both these datasets 
# separately.

# ## New Features
# Looking at the graphs, there does not seem a simple corelation between the manually identified POI's
# and the financial and email metrics
# It my be useful to create new features which may be better for analysis and prediction of POIs. 
# I'll create these features as ratios with a relevant metric. The proposed features for financial metrics are
# 1. Payment ratio = Total_payments / salary
# 2. Bonus Ratio = bonus / salary
# 3. total stock ratio = total_stock_value / salary
# 4. exercised stock ratio = exercised_stock_options / total_stock_value
# 
# The features proposed for email metrics are
# 1. poi_from_ratio = from_poi_to_this_person / from_messages
# 2. poi_to_ratio = from_this_person_to_poi / to_messages
# 3. shared_poi_ratio = shared_receipt_with_poi / from_messages
# 
# I'm hoping these ratios may be more relevant to identification of POIs'

# In[13]:



## I'm changing the entire dataframe to type float
## for convenient processing
wFinFrame = wFinFrame.astype(float)
wEmailFrame = wEmailFrame.astype(float)

## Ill remove the poi column from both frames

poi_list = wFinFrame['poi']
poi_list_e = wEmailFrame['poi']
wFinFrame.drop('poi', axis=1, inplace=True)
wEmailFrame.drop('poi', axis=1, inplace=True)

## I'll define functions that will assist in 
## creating and adding the new metrics

def divide_entries(to_divide, divide_by):
    result = 0.
    if divide_by != 0.:
        result = to_divide/divide_by
    return result

def divide_series (Series1, Series2) :
    SResult = np.vectorize(divide_entries) (Series1, Series2)
    return SResult

## I'll define separate dataframes for financial 
## and email metrics

rtioeFrame = pd.DataFrame()
rtiofFrame = pd.DataFrame()
rtiofFrame['total_payment_ratio'] = divide_series (wFinFrame['total_payments'], wFinFrame['salary'])
rtiofFrame['total_stock_ratio'] = divide_series (wFinFrame['total_stock_value'], wFinFrame['salary'])
rtiofFrame['exercised_stock_ratio'] = divide_series (wFinFrame['exercised_stock_options'], wFinFrame['total_stock_value'])
rtiofFrame['bonus_ratio'] = divide_series (wFinFrame['bonus'], wFinFrame['salary'])
rtioeFrame['poi_from_ratio'] = divide_series (wEmailFrame['from_poi_to_this_person'], wEmailFrame['to_messages'])
rtioeFrame['poi_to_ratio'] = divide_series (wEmailFrame['from_this_person_to_poi'], wEmailFrame['from_messages'])
rtioeFrame['shared_poi_ratio'] = divide_series (wEmailFrame['shared_receipt_with_poi'], wEmailFrame['to_messages'])

## print "Top Financial ratio rows", rtiofFrame.head()
## print "Top Email ratio rows", rtioeFrame.head()


# # Display of new features
# 
# I'll plot the scatter plots of the new features 
# to check on the data

# In[14]:


## Scatterplot of new features

figurenum = 1
for column in rtiofFrame:
    plt.figure(figurenum, figsize=(20,6))
    plt.scatter(rtiofFrame.index, rtiofFrame[column],label = column)
    pylab.legend(loc='upper left')
    figurenum += 1
    
for column in rtioeFrame:
    plt.figure(figurenum, figsize=(20,6))
    plt.scatter(rtioeFrame.index, rtioeFrame[column],label = column)
    pylab.legend(loc='upper left')
    figurenum += 1


# # Outliers check
# 
# Both the 'Total Payment Ratio' and 'Total Stock Ratio' features seem to
# have outliers. I'll run a check.

# In[15]:


## Checking for entries that are outliers

max_idx_payment_ratio = rtiofFrame['total_payment_ratio'].idxmax()
max_idx_stock_ratio = rtiofFrame['total_stock_ratio'].idxmax()
print 'Max payment ratio is for ', max_idx_payment_ratio
print 'Max stock ratio is for ', max_idx_stock_ratio
print "rtiofFrame", rtiofFrame.head()

max_name_payment_ratio = my_dataset.keys()[max_idx_payment_ratio]
max_name_stock_ratio = my_dataset.keys()[max_idx_stock_ratio]
print 'Max payment ratio name is ', max_name_payment_ratio
print 'Max stock ratio name is ', max_name_stock_ratio

## These entries seem to be valid, so I'll let them remain


# In[16]:


### I'll rescale the primary financial and email feature values
### My intent here is that the original values should not 
### overshadow the new features just because of large values

from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
sFinArray = scalar.fit_transform(wFinFrame)
sEmailArray = scalar.fit_transform(wEmailFrame)

sFinFrame = pd.DataFrame(data=sFinArray, index=wFinFrame.index, columns=wFinFrame.columns)
sEmailFrame = pd.DataFrame(data=sEmailArray, index=wEmailFrame.index, columns=wEmailFrame.columns)

print "scaled financial frame", sFinFrame.head()
print "scaled email frame", sEmailFrame.head()


# In[17]:


### I'll concatenate new features with the old features
### for financial data and email data separately
### I'll ignore the scaled features, since this will not impact the two algorithms I've chosen

FinFramesList = [wFinFrame, rtiofFrame]
### print "FinFramesList", FinFramesList[1:8]
all_fFrame = pd.concat(FinFramesList, axis=1)

EmailFramesList = [wEmailFrame, rtioeFrame]
all_eFrame = pd.concat(EmailFramesList, axis=1)

### print " financial frame head", all_fFrame.head(10)
### print "Email Frame Head", all_eFrame.head(10)


# In[18]:


### I'll join the new features and old features to create 
### 2 datasets of relevant features
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, chi2

### I'll initialize SelectKBest instance to reduce features
### for financial data. Will choose value of k as 3
skb = SelectKBest(chi2, 3)
print "SelectKBest value is 3"

### Ill fit the data on both the financial and email features
skb.fit(all_fFrame, poi_list)

### Print all the coefficients in skb
for ind in range(len(all_fFrame.columns)):
    print (all_fFrame.columns[ind] , skb.scores_[ind], skb.pvalues_[ind])
    
### I'll initialize SelectKBest instance to reduce features
### for financial data, k is chosen as 6
skb = SelectKBest(chi2, 6)
print "SelectKBest value is 6"

### Ill fit the data on both the financial and email features
skb.fit(all_fFrame, poi_list)

### Print all the coefficients in skb
for ind in range(len(all_fFrame.columns)):
    print (all_fFrame.columns[ind] , skb.scores_[ind], skb.pvalues_[ind])

### I'll initialize SelectKBest instance to reduce features
### for financial data. K is all.
skb = SelectKBest(chi2, k='all')
print "SelectKBest K is all"

### Ill fit the data on both the financial and email features
skb.fit(all_fFrame, poi_list)

### Print all the coefficients in skb
for ind in range(len(all_fFrame.columns)):
    print (all_fFrame.columns[ind] , skb.scores_[ind], skb.pvalues_[ind])

### I'll initialize SelectKBest instance to reduce features
### for email data
skb2 = SelectKBest(chi2, k='all')
### Ill fit the data on email features
skb2.fit(all_eFrame, poi_list_e)

### Print all the coefficients in skb
for ind in range(len(all_eFrame.columns)):
    print (all_eFrame.columns[ind] , skb.scores_[ind], skb.pvalues_[ind])

### Using pvalues less than 0.05, I found that most scores
### show corelation between the feature and the POI value.
### I'll initialize SelectKBest instance to reduce features
### for financial data. K is 11 for financial features.
skb = SelectKBest(chi2, k=7)
### Ill fit the data on both the financial and email features
skb.fit(all_fFrame, poi_list)

### For the email features, pvalues are all close to 0
### So, I'll use all the features.
### I'll initialize SelectKBest instance to reduce features
### for email data
skb2 = SelectKBest(chi2, k=7)
### Ill fit the data on email features
skb2.fit(all_eFrame, poi_list_e)
# In[19]:


### Checking the values, the top 5-6 values seem to be of importance for both financial and email data
### Will select the top 5 for each dataframe

t_fArray = skb.transform(all_fFrame)
t_eArray = skb2.transform(all_eFrame)

t_mask = skb.get_support()
e_mask = skb2.get_support()

t_fFeaNames = all_fFrame.columns[t_mask]
t_eFeaNames = all_eFrame.columns[e_mask]

t_fFrame = pd.DataFrame(data=t_fArray, columns=t_fFeaNames)
t_eFrame = pd.DataFrame(data=t_eArray, columns=t_eFeaNames)

### print "transformed financial frame ", t_fFrame.head()
### print "transformed email frame", t_eFrame.head()


# In[20]:



### Using GridSearch to fine tune parameters
### The two algorithms I'll use are GaussianNB and DecisionTrees
### For decision tree algorithm , lets find the optimal max_depth
### Here I'll only use the financial data

from sklearn import tree
from sklearn.model_selection import GridSearchCV
dt_param = {'max_depth':[3,5,9]}
dt = tree.DecisionTreeClassifier()
gs = GridSearchCV(dt, dt_param)
gs.fit(t_fFrame, poi_list)
dt_param = gs.best_params_['max_depth']
print 'dt_param', dt_param


# In[21]:



### Using GridSearch to fine tune parameters for SVM
### However, decided against using SVM 

### from sklearn import svm
### from sklearn.model_selection import GridSearchCV
### sv_param = {'kernel':["rbf","linear"], 'C':[0.001,0.01,0.1,1,10], 'gamma':[0.001,0.01,0.1,1]}
### sv = svm.SVC()
### gs = GridSearchCV(sv, sv_param)
### gs.fit(t_fFrame, poi_list)
### sv_params = gs.best_params_
### print 'sv_params', sv_params


# In[22]:


### I'll run both GaussianNB and DecisionTree on Financial data
### Based on the accuracy of each, I'll select the final 
### algorithm

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedShuffleSplit
### I'll split the data using a K-fold operation
### Lets keep 20% of data for testing.
kf = StratifiedShuffleSplit(poi_list, n_iter = 5, test_size = 0.2)
idx = 0
fold_idx=0
nbscore = []
dtscore = []
print "Fold size", kf
for training_index, testing_index in kf:
    fold_idx+=1
    training_items = pd.DataFrame()
    testing_items = pd.DataFrame()
    training_result = pd.Series()
    testing_result = pd.Series()
    for ii in training_index:
        training_items = training_items.append(t_fFrame.iloc[ii])
    for ii in testing_index:
        testing_items = testing_items.append(t_fFrame.iloc[ii])
    idx = 0
    for ii in training_index:
        training_result.set_value(idx, poi_list[ii])
        idx += 1
    idx = 0
    for ii in testing_index:
        testing_result.set_value(idx, poi_list[ii])
        idx += 1

    
###  print "Training items shape", training_index.shape, training_items.shape
###  print "Testing items shape", testing_index.shape, testing_items.shape
    
    ### Lets use Gaussian Naive Bayes on these samples
    ### Will store result in an array for all folds for 
    ### comparison later
    clf = GaussianNB()
    ###gnb  = GaussianNB()
    clf.fit(training_items, training_result)
    nbscore.append(clf.score(testing_items, testing_result))
### print "fold index", fold_idx
    
    ### Lets use DecisionTree on these samples
    ### Will store result in an array for all folds for 
    ### comparison later
    dtc = tree.DecisionTreeClassifier(max_depth=3)
    dtc.fit(training_items, training_result)
    dtscore.append(dtc.score(testing_items, testing_result))
    
   


# ### Choice of algorithm
# 
# Comparing the accuracy values, it seems both algorithms are fairly close. After multiple tries, 
# I decided to choose the Naive Bayes algorithm, since it seems to have slightly better results.
# I'll go ahead with it to validate the results.
# 

# In[23]:


### I'll check precision and recall values for GaussianNB
### I'll split the data using a K-fold operation
### Lets keep 25% of data for testing.
from sklearn.metrics import precision_recall_fscore_support
kf = StratifiedShuffleSplit(poi_list, n_iter = 4, test_size = 0.25)
idx = 0
fold_idx=0
dtscore = []
testing_predict = pd.Series()
print "Fold size", kf
for training_index, testing_index in kf:
    fold_idx+=1
    training_items = pd.DataFrame()
    testing_items = pd.DataFrame()
    training_result = pd.Series()
    testing_result = pd.Series()
    for ii in training_index:
        training_items = training_items.append(t_fFrame.iloc[ii])
    for ii in testing_index:
        testing_items = testing_items.append(t_fFrame.iloc[ii])
    idx = 0
    for ii in training_index:
        training_result.set_value(idx, poi_list[ii])
        idx += 1
    idx = 0
    for ii in testing_index:
        testing_result.set_value(idx, poi_list[ii])
        idx += 1
    
   
    ### print "Training items shape", training_index.shape, training_items.shape
    ### print "Testing items shape", testing_index.shape, testing_items.shape
    
    ### Lets use Gaussian Naive Bayes on these samples
    ### Will store result in an array for all folds for 
    ### comparison later
    dtc = GaussianNB()
    dtc.fit(training_items, training_result)
    testing_predict = dtc.predict(testing_items)
    prec, recal, fscor, supt = precision_recall_fscore_support(testing_result, testing_predict)
    ### print "test result is ", testing_result
    ### print "test prediction is", testing_predict
    print "Precision values for fold ", fold_idx, "are ", prec
    print "Recall values for fold", fold_idx, "are ", recal


# In[24]:


### Let's try the same algorithm on email data
### I'll split the data using a K-fold operation
### Lets keep 25% of data for testing.
from sklearn.metrics import precision_recall_fscore_support
kf = StratifiedShuffleSplit(poi_list_e, n_iter = 4, test_size = 0.25)
idx = 0
fold_idx=0
dtscore = []
testing_predict = pd.Series()
print "Fold size", kf
for training_index, testing_index in kf:
    fold_idx+=1
    training_items = pd.DataFrame()
    testing_items = pd.DataFrame()
    training_result1 = pd.Series()
###    print "training_result1", training_result1
    testing_result1 = pd.Series()
###    print "testing_result1", testing_result1
    for ii in training_index:
        training_items = training_items.append(t_eFrame.iloc[ii])
    for ii in testing_index:
        testing_items = testing_items.append(t_eFrame.iloc[ii])
    idx = 0
    for ii in training_index:
###        print "value of ii", ii
###        print "value of poi_list[ii]", poi_list[ii]
        training_result1.reset_index()
        training_result1.set_value(idx, poi_list_e[ii])
        idx += 1
    idx = 0
    for ii in testing_index:
        testing_result1.reset_index()
        testing_result1.set_value(idx,poi_list_e[ii])
        idx += 1
    
   
    ### print "Training items shape", training_index.shape, training_items.shape
    ### print "Training items after selectbest", len(train_fin_top2)
    ### print "Testing items shape", testing_index.shape, testing_items.shape
    ### print "Testing items after selectbest", len(test_fin_top2)
    
    ### Lets use Gaussian Naive Bayes on these samples
    ### Will store result in an array for all folds for 
    ### comparison later
    ### clf = GaussianNB()


    clf = tree.DecisionTreeClassifier(max_depth=7)
    clf.fit(training_items, training_result1)
    testing_predict = clf.predict(testing_items)
    prec, recal, fscor, supt = precision_recall_fscore_support(testing_result1, testing_predict)
    ### print "test result is ", testing_result1
    ### print "test prediction is", testing_predict
    print "Precision values for fold ", fold_idx, "are ", prec
    print "Recall values for fold", fold_idx, "are ", recal


# ### Results of Performance metrics
# 
# The performance metrics are evaluated separately for POIs and non-POIs
# The metrics are better for financial data than for email data
# Metrics for non-POIs are much higher than that for POIs. In case of 
# precision, the values for non POI's is close to 0.75, while that for POI's varies
# from 1.0 to 0 depending on the test data.
# The Recall values similarly are closer to 0.7 for non POIs, but vary from 0. to 1. 
# for POIs. 
# For submission I'll include the GaussianNB algorithm with the final
# financial dataset I used for my testing.

# In[25]:
### Lets change the financial data frame to a dictionary
### before dumping it out
t_fFrame['poi'] = poi_list
column_names = t_fFrame.columns.values.tolist()
column_names.remove('poi')
column_names.insert(0,'poi')
t_fDict = t_fFrame.T.to_dict('dict')
print "t_fDict head", t_fDict.items()[0:5]
print "feature names", column_names

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, t_fDict, column_names)
