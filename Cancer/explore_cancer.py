'''The main module running the different '''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import models
import preprocessing as prepro 


def main():
	cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")
	
	target_threshold = 32
	cancer["survival"] = ""	
 
	cancer = prepro.setup_target_variable(cancer, 'OS MONTHS', 'survival', 32)

	cancer = prepro.handle_categorical_variables(cancer, ['AJCC METASTASIS PATHOLOGIC PM', 'AJCC NODES PATHOLOGIC PN', 'AJCC PATHOLOGIC TUMOR STAGE', 'AJCC TUMOR PATHOLOGIC PT', 'DFS STATUS', 'ETHNICITY', 'GENDER'])

	cancer = prepro.handle_na(cancer)

	print "Shape(Before removing NaN): ", cancer.shape
	cancer_without_nan = cancer.dropna(subset = ['OS MONTHS'])
	print "Shape(After removing NaN): ", cancer_without_nan.shape

	models.DecisionTree(cancer_without_nan, [2])	

if __name__ == '__main__':
	main()