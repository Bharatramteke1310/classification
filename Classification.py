import pandas as pd
import numpy as np
import os

dataset_Train = pd.read_csv('D:\cognitior\Basics of data science\\Python_Module_Day_15.2_Credit_Risk_Train_data_XTRAIN.csv')
dataset_New_Pred = pd.read_csv('D:\\cognitior\\Basics of data science\\Python_Module_Day_15.3_Credit_Risk_Test_data.csv')
dataset_Test = pd.read_csv('D:\\cognitior\\Basics of data science\\Python_Module_Day_15.4_Credit_Risk_Validate_data_XTEST.csv')

dataset_Train

Q1 = dataset_Train.quantile(0.25)
Q3 = dataset_Train.quantile(0.75)
IQR = Q3-Q1
​
dataset_Train = dataset_Train[~((dataset_Train<(Q1-1.5*IQR))
                       | (dataset_Train>(Q3+1.5*IQR))).any(axis=1)]
​
dataset_Train

Q1 = dataset_Test.quantile(0.25)
Q3 = dataset_Test.quantile(0.75)
IQR = Q3-Q1

dataset_Test = dataset_Test[~((dataset_Test<(Q1-1.5*IQR))
                       | (dataset_Test>(Q3+1.5*IQR))).any(axis=1)]

dataset_Test

dataset_Train.isnull().sum()

dataset_Train["LoanAmount"].fillna(dataset_Train["LoanAmount"].mean(), inplace=True)

dataset_Train["Loan_Amount_Term"].fillna(dataset_Train["Loan_Amount_Term"].mean(), inplace=True)
dataset_Train["Credit_History"].fillna(dataset_Train["Credit_History"].mean(), inplace=True)

dataset_Train.groupby('Dependents').agg({'Dependents': np.size})
dataset_Train["Dependents"]=dataset_Train["Dependents"].fillna('0')
dataset_Train["Dependents"] = dataset_Train["Dependents"].astype('category')

dataset_Train.groupby('Gender').agg({'Gender': np.size})
dataset_Train["Gender"]=dataset_Train["Gender"].fillna('Male')
dataset_Train["Gender"] = dataset_Train["Gender"].astype('category')

dataset_Train.groupby('Married').agg({'Married': np.size})
dataset_Train["Married"]=dataset_Train["Married"].fillna('Yes')
dataset_Train["Married"] = dataset_Train["Married"].astype('category')

dataset_Train.groupby('Self_Employed').agg({'Self_Employed': np.size})
dataset_Train["Self_Employed"]=dataset_Train["Self_Employed"].fillna('No')
dataset_Train["Self_Employed"] = dataset_Train["Self_Employed"].astype('category')

dataset_Train = pd.get_dummies(dataset_Train, columns=['Dependents'], drop_first=True)
dataset_Train = pd.get_dummies(dataset_Train, columns=['Gender'], drop_first=True)
dataset_Train = pd.get_dummies(dataset_Train, columns=['Married'], drop_first=True)
dataset_Train = pd.get_dummies(dataset_Train, columns=['Property_Area'], drop_first=True)

dataset_Train = pd.get_dummies(dataset_Train, columns=['Education'], drop_first=True)

dataset_Train = pd.get_dummies(dataset_Train, columns=['Self_Employed'], drop_first=True)
dataset_Train = pd.get_dummies(dataset_Train, columns=['Loan_Status'], drop_first=True)
dataset_Train

dataset_Train_d=dataset_Train.drop(['Loan_ID'], axis=1)

x_train = dataset_Train_d.iloc[:,:-1].values
pd.DataFrame(x_train)

y_train = dataset_Train_d.iloc[:,-1:].values
y_train

#Test DataSet
dataset_Test.isnull().sum()

dataset_Test

dataset_Test["LoanAmount"].fillna(dataset_Test["LoanAmount"].mean(), inplace=True)
dataset_Test["Loan_Amount_Term"].fillna(dataset_Test["Loan_Amount_Term"].mean(), inplace=True)
dataset_Test["Credit_History"].fillna(dataset_Test["Credit_History"].mean(), inplace=True)

dataset_Test.groupby('Dependents').agg({'Dependents': np.size})
dataset_Test["Dependents"]=dataset_Test["Dependents"].fillna('0')
dataset_Test["Dependents"] = dataset_Test["Dependents"].astype('category')

dataset_Test.groupby('Gender').agg({'Gender': np.size})
dataset_Test["Gender"]=dataset_Test["Gender"].fillna('Male')
dataset_Test["Gender"] = dataset_Test["Gender"].astype('category')

dataset_Test.groupby('Self_Employed').agg({'Self_Employed': np.size})
dataset_Test["Self_Employed"]=dataset_Test["Self_Employed"].fillna('No')
dataset_Test["Self_Employed"] = dataset_Test["Self_Employed"].astype('category')

dataset_Test.isnull().sum()

dataset_Test = pd.get_dummies(dataset_Test, columns=['Dependents'], drop_first=True)
dataset_Test = pd.get_dummies(dataset_Test, columns=['Gender'], drop_first=True)
dataset_Test = pd.get_dummies(dataset_Test, columns=['Married'], drop_first=True)
dataset_Test = pd.get_dummies(dataset_Test, columns=['Property_Area'], drop_first=True)
dataset_Test = pd.get_dummies(dataset_Test, columns=['Education'], drop_first=True)
dataset_Test = pd.get_dummies(dataset_Test, columns=['Self_Employed'], drop_first=True)

dataset_Test = pd.get_dummies(dataset_Test, columns=['outcome'], drop_first=True)

dataset_Test

dataset_Test_d=dataset_Test.drop(['Loan_ID'], axis=1)

x_test = dataset_Test_d.iloc[:,:-1].values
pd.DataFrame(x_test)

y_test = dataset_Test_d.iloc[:,-1:].values
pd.DataFrame(y_test)

from sklearn.preprocessing import StandardScaler
sc_x_test = StandardScaler()
x_test = sc_x_test.fit_transform(x_test)

pd.DataFrame(x_test)

dataset_New_Pred
Q1 = dataset_New_Pred.quantile(0.25)
Q3 = dataset_New_Pred.quantile(0.75)
IQR = Q3-Q1

dataset_New_Pred = dataset_New_Pred[~((dataset_New_Pred<(Q1-1.5*IQR))
                       | (dataset_New_Pred>(Q3+1.5*IQR))).any(axis=1)]

dataset_New_Pred.isnull().sum()

dataset_New_Pred["LoanAmount"].fillna(dataset_New_Pred["LoanAmount"].mean(), inplace=True)
dataset_New_Pred["Loan_Amount_Term"].fillna(dataset_New_Pred["Loan_Amount_Term"].mean(), inplace=True)
dataset_New_Pred["Credit_History"].fillna(dataset_New_Pred["Credit_History"].mean(), inplace=True)

dataset_New_Pred.groupby('Dependents').agg({'Dependents': np.size})
dataset_New_Pred["Dependents"]=dataset_New_Pred["Dependents"].fillna('0')
dataset_New_Pred["Dependents"] = dataset_New_Pred["Dependents"].astype('category')

dataset_New_Pred.groupby('Gender').agg({'Gender': np.size})
dataset_New_Pred["Gender"]=dataset_New_Pred["Gender"].fillna('Male')
dataset_New_Pred["Gender"] = dataset_New_Pred["Gender"].astype('category')

dataset_New_Pred.groupby('Self_Employed').agg({'Self_Employed': np.size})
dataset_New_Pred["Self_Employed"]=dataset_New_Pred["Self_Employed"].fillna('No')
dataset_New_Pred["Self_Employed"] = dataset_New_Pred["Self_Employed"].astype('category')

dataset_New_Pred_d = dataset_New_Pred.drop(['Loan_ID'], axis=1)

dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Dependents'], drop_first=True)
dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Gender'], drop_first=True)
dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Married'], drop_first=True)
dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Property_Area'], drop_first=True)
dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Education'], drop_first=True)
dataset_New_Pred_d = pd.get_dummies(dataset_New_Pred_d, columns=['Self_Employed'], drop_first=True)

len(dataset_New_Pred_d)

x_new= dataset_New_Pred_d.iloc[:,:].values
x_new

sc_x_new= StandardScaler()
x_new = sc_x_new.fit_transform(x_new)
x_new

pd.DataFrame(x_new)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

y_pred = logmodel.predict(x_new)
y_pred

len(y_pred)

len(y_test)

y_pred = logmodel.predict(x_new)

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5, 
                                      metric='minkowski', p=2)

classifier_knn.fit(x_train, y_train)

y_pred = classifier_knn.predict(x_test)
y_pred

confusion_matrix(y_test, y_pred)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train, y_train)
y_pred=classifier_nb.predict(x_test)

y_pred

confusion_matrix(y_test, y_pred)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


#sigmoid
from sklearn.svm import SVC
classifier_svm = SVC(kernel='sigmoid')
classifier_svm.fit(x_train,y_train)


#Linear
from sklearn.svm import SVC
classifier_svm = SVC(kernel='linear')
classifier_svm.fit(x_train,y_train)

y_pred = classifier_svm.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion='entropy')
classifier_dt.fit(x_train,y_train)

y_pred = classifier_dt.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=3, criterion='entropy')
classifier_rf.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)



