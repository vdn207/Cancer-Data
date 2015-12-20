import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn
from sklearn import linear_model, datasets, svm
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import *
import math
from collections import OrderedDict
import preprocessing as prepro

variablesOfInterest = ['AGE','AJCC METASTASIS PATHOLOGIC PM','AJCC NODES PATHOLOGIC PN','AJCC PATHOLOGIC TUMOR STAGE','AJCC TUMOR PATHOLOGIC PT','ETHNICITY','GENDER','INITIAL WEIGHT','Mutation Count','DFS STATUS']

cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")

target_threshold = 32
cancer["survival"] = ""	

#print cancer.isnull().sum()
cancer['OS MONTHS'].fillna(cancer['OS MONTHS'].mean, inplace = True)    #need to fill missing values before manufactoring variable
cancer = prepro.setup_target_variable(cancer, 'OS MONTHS', 'survival', 32)

cancer = cancer[variablesOfInterest + [ "survival"] ]

cancer = prepro.handle_categorical_variables(cancer, ['AJCC METASTASIS PATHOLOGIC PM', 'AJCC NODES PATHOLOGIC PN', 'AJCC PATHOLOGIC TUMOR STAGE', 'AJCC TUMOR PATHOLOGIC PT', 'DFS STATUS', 'ETHNICITY', 'GENDER'])

cancer = prepro.handle_na(cancer)



#seperate into test and train
n_row = cancer.shape[0]
print n_row

cutIndex = int(n_row *.8)
train_df , test_df = cancer[:cutIndex] , cancer[cutIndex:]
train_target=train_df['survival']
test_target=test_df['survival']

train_df.drop('survival',1)
test_df.drop('survival',1)

train_df= train_df.convert_objects( convert_numeric =True)
train_target= train_target.convert_objects( convert_numeric =True)
print train_target[:7]

#Logistic regression model creation and fitting
my_log_reg = linear_model.LogisticRegression(C=1e30)
my_log_reg1 = my_log_reg.fit(train_df,train_target )

#SVM model creation and fitting
my_svm = svm.SVC(kernel='linear',probability = True)
my_svm1 = my_svm.fit(train_df,train_target)

#Next two lines plots the random line
#x=[0.0,1.0]
#plt.plot(x, x, linestyle ='dashed',color ='red',linewidth =2, label = 'Random line')

#Plotting SVM
preds = my_svm1.predict_proba(test_df)[:,1]
fpr, tpr, _ = metrics.roc_curve(test_target, preds)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.plot(fpr,tpr, label = 'SVM with AUC={0:.3f}'.format(metrics.auc(fpr,tpr)))


#Plotting Logisitic regression
preds = my_log_reg1.predict_proba(test_df)[:,1]
fpr, tpr, _ = metrics.roc_curve(test_target, preds)

auc_log_reg = metrics.auc(fpr,tpr)

plt.plot(fpr,tpr, label = 'LogReg with AUC={0:.3f}'.format(metrics.auc(fpr,tpr)))
plt.fill_between(fpr,tpr, alpha = 0.2)
plt.legend(loc =2,bbox_to_anchor=(1.05,1))


