import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


def convert_categorical_to_int(df, column_name):
	categorical_values = df[column_name].value_counts().index.tolist()
	assigned_value = 0
	for value in categorical_values:
		df.loc[df[column_name] == value, column_name] = assigned_value
		assigned_value += 1

	return df

def main():
	cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")
	#os_months = cancer['OS MONTHS'].values
	k = 32

	cancer["survival"] = ""	
	#print cancer_without_nan.shape
 
	# Encoding the number of OS MONTHS into 1's and 0's
	cancer.loc[cancer['OS MONTHS'].values > k, 'survival'] = 1
	cancer.loc[cancer['OS MONTHS'].values <= k, 'survival'] = 0

	'''Replacing Categorical variables with Integers'''
	# Setting up AJCC METASTASIS PATHOLOGIC PM values - Index = 3
	cancer = convert_categorical_to_int(cancer, 'AJCC METASTASIS PATHOLOGIC PM')
	
	# Setting up AJCC NODES PATHOLOGIC PN values - Index = 4
	cancer = convert_categorical_to_int(cancer, 'AJCC NODES PATHOLOGIC PN')

	# Setting up AJCC PATHOLOGIC TUMOR STAGE values - Index = 5
	cancer = convert_categorical_to_int(cancer, 'AJCC PATHOLOGIC TUMOR STAGE')

	# Setting up AJCC TUMOR PATHOLOGIC PT values - Index = 7
	cancer = convert_categorical_to_int(cancer, 'AJCC TUMOR PATHOLOGIC PT')

	# Setting up DFS STATUS values - Index = 16
	cancer = convert_categorical_to_int(cancer, 'DFS STATUS')

	# Setting up ETHICITY values - Index = 17
	cancer = convert_categorical_to_int(cancer, 'ETHNICITY')

	# Setting up the GENDER values - Index = 19
	cancer = convert_categorical_to_int(cancer, 'GENDER')

	'''Replacing NaN values in each column'''

	cancer['AGE'].fillna(cancer['AGE'].mean, inplace = True)
	cancer['AJCC METASTASIS PATHOLOGIC PM'].fillna(cancer['AJCC METASTASIS PATHOLOGIC PM'].value_counts()[0], inplace = True)
	cancer['AJCC NODES PATHOLOGIC PN'].fillna(cancer['AJCC NODES PATHOLOGIC PN'].value_counts()[0], inplace = True)
	cancer['AJCC PATHOLOGIC TUMOR STAGE'].fillna(cancer['AJCC PATHOLOGIC TUMOR STAGE'].value_counts()[0], inplace = True)
	cancer['AJCC TUMOR PATHOLOGIC PT'].fillna(cancer['AJCC TUMOR PATHOLOGIC PT'].value_counts()[0], inplace = True)
	cancer['DFS STATUS'].fillna(cancer['DFS STATUS'].value_counts()[0], inplace = True)
	cancer['ETHNICITY'].fillna(cancer['ETHNICITY'].value_counts()[0], inplace = True)
	cancer['GENDER'].fillna(cancer['GENDER'].value_counts()[0], inplace = True)

	#print type_of_target(cancer['survival'])

	print "Shape(Before removing NaN): ", cancer.shape
	cancer_without_nan = cancer.dropna(subset = ['OS MONTHS'])
	print "Shape(After removing NaN): ", cancer_without_nan.shape

	# Split the training and test data
	cancer_train, cancer_test, survival_train, survival_test = train_test_split(cancer_without_nan.drop('survival', 1), cancer_without_nan.survival)
	
	# Training the data using Decision Tree
	decision_tree = DecisionTreeClassifier(criterion = 'entropy')
	decision_tree.fit(cancer_train.iloc[:, [2, 3, 4, 5, 7, 16, 17, 19]], survival_train.astype(int))

	print "Features Used:", cancer_without_nan.iloc[:, [2, 3, 4, 5, 7, 16, 17, 19]].columns.values 
	
	# Testing on the training data
	print "Accuracy (Training Data): ", decision_tree.score(cancer_train.iloc[:, [2, 3, 4, 5, 7, 16, 17, 19]], survival_train.astype(int))

	# Testing the data
	print "Accuracy (Test Data): ", decision_tree.score(cancer_test.iloc[:, [2, 3, 4, 5, 7, 16, 17, 19]], survival_test.astype(int))
	

if __name__ == '__main__':
	main()