'''Contains functions to clean/pre-process the data'''

import numpy as np 
import pandas as pd 


variablesOfInterest = ['AGE','AJCC METASTASIS PATHOLOGIC PM','AJCC NODES PATHOLOGIC PN','AJCC PATHOLOGIC TUMOR STAGE','AJCC TUMOR PATHOLOGIC PT','ETHNICITY','GENDER','INITIAL WEIGHT','Mutation Count','DFS STATUS']


def convert_categorical_to_int(df, column_name):
	'''Convert categorical values into integers'''	

	categorical_values = df[column_name].value_counts().index.tolist()
	assigned_value = 0
	for value in categorical_values:
		df.loc[df[column_name] == value, column_name] = assigned_value
		assigned_value += 1

	return df

def handle_categorical_variables(df, list_of_columns):
	'''Converts the categorical variables in the list of columns to integers'''

	for column in list_of_columns:
		df = convert_categorical_to_int(df, column)

	return df

def handle_na(df, ):
	'''Handle the NaN fields'''

	df['AGE'].fillna(df['AGE'].mean, inplace = True)
	df['AJCC METASTASIS PATHOLOGIC PM'].fillna(df['AJCC METASTASIS PATHOLOGIC PM'].value_counts()[0], inplace = True)
	df['AJCC NODES PATHOLOGIC PN'].fillna(df['AJCC NODES PATHOLOGIC PN'].value_counts()[0], inplace = True)
	df['AJCC PATHOLOGIC TUMOR STAGE'].fillna(df['AJCC PATHOLOGIC TUMOR STAGE'].value_counts()[0], inplace = True)
	df['AJCC TUMOR PATHOLOGIC PT'].fillna(df['AJCC TUMOR PATHOLOGIC PT'].value_counts()[0], inplace = True)
	df['DFS STATUS'].fillna(df['DFS STATUS'].value_counts()[0], inplace = True)
	df['ETHNICITY'].fillna(df['ETHNICITY'].value_counts()[0], inplace = True)
	df['GENDER'].fillna(df['GENDER'].value_counts()[0], inplace = True)
	df['INITIAL WEIGHT'].fillna(df['INITIAL WEIGHT'].mean, inplace = True)
	df['Mutation Count'].fillna(df['Mutation Count'].mean, inplace = True)

	return df 

def setup_target_variable(df, target_column, new_column, threshold):
	'''Changes the target variable values based on the threshold'''

	df.loc[df[target_column].values > threshold, new_column] = 1		# 1 - above the threshold
	df.loc[df[target_column].values <= threshold, new_column] = 0		# 0 - below the threshold

	return df 





