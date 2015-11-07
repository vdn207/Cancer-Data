'''Different prediction models used on the data'''

import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


def DecisionTree(sample_data, features_list, plot=False):
	'''Implements the decision tree classifier'''

	training_data, test_data, target_train, target_test = train_test_split(sample_data.drop('survival', 1), sample_data.survival)

	decision_tree = DecisionTreeClassifier(criterion = 'entropy')
	decision_tree.fit(training_data.iloc[:, features_list], target_train.astype(int))

	print "Features Used:", sample_data.iloc[:, features_list].columns.values 
	print "-----------------------------------------------------------------"
	print "Feature Importances:", list(decision_tree.feature_importances_)
	# Testing on the training data
	print "Accuracy (Training Data): ", decision_tree.score(training_data.iloc[:, features_list], target_train.astype(int))

	# Testing the data
	print "Accuracy (Test Data): ", decision_tree.score(test_data.iloc[:, features_list], target_test.astype(int))

	'''
	if plot:
		plt.barh(np.arange(1, 2 * len(features_list), 2), list(decision_tree.feature_importances_))
		plt.xlabel('Feature Importance')
		plt.title('Bar Plot of Feature Importance Provided by Decision Tree')
		--> Error #plt.yticks(np.arange(1, 2 * len(features_list), 2) + 0.25, list(sample_data.iloc[:, sample_data.columns.values))
		plt.savefig("Decision_Tree_Features_Importances")
	'''