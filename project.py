# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:01:22 2018

@author: dilip
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import scikitplot as skplt


# Loading the dataset
dataset = pd.read_csv('loan.csv')
dataset.head()


#Checking types of loan status
dataset['loan_status'].value_counts()

#Only Fully Paid and Charged Off are needed
x = dataset[dataset.loan_status =="Fully Paid"]
y = dataset[dataset.loan_status =="Charged Off"]
dataset = x.append(y)

#Removing irrelavant , redundant columns
dataset.drop('id',axis=1,inplace=True)
dataset.drop('member_id',axis=1,inplace=True)
dataset.drop('emp_title',axis=1,inplace=True)
dataset.drop('zip_code',axis=1,inplace=True)
dataset.drop('title',axis=1,inplace=True)
dataset.drop('out_prncp',axis=1,inplace=True)
dataset.drop('out_prncp_inv',axis=1,inplace=True)
dataset.drop('total_pymnt',axis=1,inplace=True)
dataset.drop('total_pymnt_inv',axis=1,inplace=True)
dataset.drop('last_pymnt_d',axis=1,inplace=True)
dataset.drop('last_pymnt_amnt',axis=1,inplace=True)
dataset.drop('next_pymnt_d',axis=1,inplace=True)
dataset.drop('total_rec_late_fee',axis=1,inplace=True)
dataset.drop('total_rec_int',axis=1,inplace=True)
dataset.drop('issue_d',axis=1,inplace=True)
dataset.drop('url',axis=1,inplace=True)
dataset.drop('recoveries',axis=1,inplace=True)
dataset.drop('collection_recovery_fee',axis=1,inplace=True)
dataset.drop('total_rec_prncp',axis=1,inplace=True)


#deleting columns that never change values
dataset = dataset.loc[:,dataset.apply(pd.Series.nunique) != 1]
                      
#Getting Discrete columns
discrete_columns=[]
for column in dataset.columns:
    if str(dataset[column].dtype)=="object":
        discrete_columns.append(column)

#Getting unique counts of Discrete columns        
unique_value_counts=dataset[discrete_columns].apply(lambda x:(x.unique().shape[0]),axis=0).values
unique_value_counts=pd.DataFrame({"column":discrete_columns,"unique_count":unique_value_counts})
unique_value_counts

#Deleting columns with more than 30 categories
cols=unique_value_counts[unique_value_counts['unique_count']>30]['column']
for col in cols:
    dataset.drop(col,axis=1,inplace=True)
    
#deleting columns which change value only once 
dataset.drop('pymnt_plan',axis=1,inplace=True)
dataset.drop('application_type',axis=1,inplace=True)
    
#deleting columns that have more than 30% as null    
columns=dataset.columns
missing_counts=dataset.apply(lambda x:np.sum(x.isnull()),axis=0).values
missing_counts_table=pd.DataFrame({"column":columns,"missing_count":missing_counts})
higher_missing_counts=missing_counts_table[missing_counts_table['missing_count']>0.2*dataset.shape[0]]
dataset.drop(higher_missing_counts['column'].values,axis=1,inplace=True)


#finding number of missing count for remaining columns which needs to be imputed
missing_counts_table=missing_counts_table[missing_counts_table['missing_count']<0.2*dataset.shape[0]]
missing_counts_table

#emp_length preprocessing to remove string
emp_length= dataset['emp_length'].str.extract('(\d+)').astype(float)
dataset.drop('emp_length',axis=1,inplace=True)
dataset['emp_length'] = emp_length

#imputing employee length with 0
dataset['emp_length'].fillna(0,inplace=True)

#histogram for data distribution
plt.figure(figsize=(20,15))
plt.subplot(311)
plt.hist(dataset['revol_util'].dropna(),bins=100);
plt.title("revol_util")
plt.subplot(312)
plt.hist(dataset['collections_12_mths_ex_med'].dropna(),bins=100);
plt.title("collections_12_mths_ex_med")


#imputing revol_util with mean
dataset['revol_util'].fillna(dataset['revol_util'].mean(),inplace=True)

#imputing collections_12_mths_ex_med with median
dataset['collections_12_mths_ex_med'].fillna(dataset['collections_12_mths_ex_med'].median(),inplace=True)

#Pair wise corelation
data_for_correlation=dataset.copy()
columns = dataset.columns
for column in columns:
    if str(data_for_correlation[column].dtype)=="object":
        data_for_correlation.drop(column,inplace=True, axis=1)

#plotting the heatmap for corelation        
corr=np.array(data_for_correlation.corr())
plt.figure(figsize=(20,30))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale=1.4)
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,annot=True,fmt=".2f",annot_kws={"size": 12},
                     yticklabels=data_for_correlation.columns,xticklabels=data_for_correlation.columns,cbar_kws={"shrink": .52})


#dropping installemnt,funded_amnt,funded_amnt_inv as it is highly corelated with loan_amount
dataset.drop('installment',axis=1,inplace=True)
dataset.drop('funded_amnt',axis=1,inplace=True)
dataset.drop('funded_amnt_inv',axis=1,inplace=True)


#moving the outcome variable to last
loan_status= dataset['loan_status']
dataset.drop('loan_status',axis=1,inplace=True)
dataset['loan_status'] = loan_status


#Encoding grade of loan (Ordinal)
dataset['grade'].value_counts()
cleanup_nums = {"grade":{"A": 1, "B": 2,"C": 3, "D": 4,"E": 5, "F": 6,"G": 7}}
dataset.replace(cleanup_nums, inplace=True)


# Encoding other categorical data
dataset = dataset.apply(LabelEncoder().fit_transform)

# diving the dataset based on the loan status
df_majority = dataset[dataset.loan_status==1]
df_minority = dataset[dataset.loan_status==0]
n =len(df_majority.index)
m =len(df_minority.index)
print()
print("Instances belonging to good Loan:",n)
print("Instances belonging to bad Loan:",m)


#Upsampling majority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=n, random_state=123) 
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


#Seperating depenedent and independent variables
X = df_upsampled.iloc[:, :-1].values
y = df_upsampled.iloc[:, -1].values


#Applying one Hot encoder to the dependent categorical variables
onehotencoder = OneHotEncoder(categorical_features = [1,4,6,7,16])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 456)


# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Naive Bayes
# we create an instance of NB Classifier and fit the data.
nb_classifier =  BernoulliNB()
nb_classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nb_classifier.predict(X_test)
print()
print("f1_score for Naive Bayes is: ")
print(f1_score(y_test, y_pred))
print("Accuracy for Naive Bayes is: ")
print(accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))



#Logistic
# we create an instance of Logistic Regression and fit the data.
regressor = LogisticRegression()
regressor.fit(X_train,y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
print()
print("f1_score for Logistic Regression is: ")
print(f1_score(y_test, y_pred))
print("Accuracy for Logistic Regression is: ")
print(accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))




#MLP NN
# we create an instance of MLP Classifer and fit the data.
nn_classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=True)
nn_classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nn_classifier.predict(X_test)
print()
print("f1_score for MLP Neural Network  is: ")
print(f1_score(y_test, y_pred))
print("Accuracy for MLP Neural Network is: ")
print(accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))



#Ensemble 
# create the sub models
estimators = []
model1 = LogisticRegression(C=1e5)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = MLPClassifier()
estimators.append(('mlp', model3))
#create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,y_train)
# Predicting the Test set results
y_pred = ensemble.predict(X_test)
print()
print("f1_score for Voting Classifier is: ")
print(f1_score(y_test, y_pred))
print("Accuracy for Voting Classifier is: ")
print(accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))




## Random Forest
classifier = RandomForestClassifier(max_depth=100,min_samples_split=20,n_estimators=100)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print()
print("f1_score for Random Forest is: ")
print(f1_score(y_test, y_pred))
print("Accuracy for Random Forest is: ")
print(accuracy_score(y_test, y_pred))
print()
print(confusion_matrix(y_test,y_pred))
print()
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))



#ROC Curve for random forest model 
prob = classifier.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, prob)
plt.show()