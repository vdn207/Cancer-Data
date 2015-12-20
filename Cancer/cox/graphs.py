import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")
ages = cancer["AGE"]
ages = ages.dropna()
ages = ages.astype(int)

overallSurvival = cancer["OS MONTHS"]
overallSurvival = overallSurvival.dropna()
overallSurvival = overallSurvival.astype(int)



ageCounts =  np.bincount(ages)
minAge = min(ages)
maxAge= max(ages)


plt.hist( range(minAge,maxAge+1), weights = ageCounts[minAge:], bins = (maxAge-minAge)/3)
plt.xlabel('Age Of Cancer Patients' )
plt.ylabel('Count')
plt.title('Age Counts for Breast Cancer Patients')
plt.show()

def histogramCancerData(variableOfInterest):
	
	histData = cancer[variableOfInterest]
	histData = histData.dropna()
	histData = histData.astype(int)

	threshold =24
	varX = histData['OS MONTHS']
	varY= histData['OS STATUS']
	
	mask = histData['OS MONTHS'] < threshold & histData['OS STATUS']==DECEASED
	print mask
	
	
'''	
	varCounts =  np.bincount(histData)
	minVar = min(histData)
	maxVar= max(histData)


	plt.hist( range(minVar,maxVar+1), weights = varCounts[minVar:], bins = (maxVar-minVar)/3)
	plt.xlabel(str(variableOfInterest ))
	plt.ylabel('Count')
	plt.title(' Counts for Breast Cancer Patients for ' +str(variableOfInterest) )
	plt.show()
'''



histogramCancerData("OS MONTHS")

