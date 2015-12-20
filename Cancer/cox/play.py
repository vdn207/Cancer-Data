import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")

varX = cancer['OS MONTHS']
varY= cancer['OS STATUS']
alive = cancer[ cancer['OS STATUS']=="LIVING"]
dead = cancer[ cancer['OS STATUS']=="DECEASED"]


threshold = 30
'''
cancer=cancer[~ np.logical_and( cancer['OS MONTHS'] < threshold , cancer['OS STATUS'] == "LIVING" ) ]
print "num in study", cancer.shape[0]
cancer['target1'] = np.logical_and( cancer['OS MONTHS'] < threshold , cancer['OS STATUS'] == "DECEASED" )

print "died " , cancer['target1'].sum() 

'''



deadBeforeThreshold = np.logical_and( varX < threshold , varY == "DECEASED" ).sum()
#print deadBeforeThreshold



censored = np.logical_and( varX < threshold , varY == "LIVING" ).sum()
#print censored

percentDeadUncensored=[]
percentDeadCensored=[]
numDead=[]
numCensored=[]

numberAtRisk=[]
NumberOfDeaths=[]
NumberCensored=[]
SurvivalProbability=[]

for threshold in range(0,100):
	numberAtRisk.append( (cancer['OS MONTHS'] >= threshold  ).sum()  )
	NumberOfDeaths.append(   np.logical_and(dead['OS MONTHS'] > threshold -1 , dead['OS MONTHS']<threshold  ).sum()   )
	NumberCensored.append(   np.logical_and(alive['OS MONTHS'] > threshold -1 , alive['OS MONTHS'] < threshold  ).sum()   )

print "numberAtRisk " , numberAtRisk

survival=[1]	
for threshold in range(1,99):
	survival.append( survival[threshold -1] - NumberOfDeaths[threshold]/float(numberAtRisk[threshold] )    ) 

print "survival " , survival

for threshold in range(1,100):
	numDead.append( np.logical_and( varX < threshold , varY == "DECEASED" ).sum()  )
	numCensored.append( np.logical_and( varX < threshold , varY == "LIVING" ).sum()  )
	
for i in range(len(numDead)):
	percentDeadCensored.append ( 1- numDead[i]/(1105.0 - numCensored[i] )   )
	percentDeadUncensored.append (1 - numDead[i]/(1105.0  )   )


	
'''

plt.plot(range(60), percentDeadCensored[:60] , label ='Censored Taken Out of Dataset')
plt.plot(range(60), percentDeadUncensored[:60] , label = 'Censored Kept in Dataset')
plt.plot(range(60), survival[0:60] , label = 'Kaplan-Meier Approach')
plt.title("Effects of Censoring based on threshold", fontsize=20)

legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.xlabel('months',fontsize = 15)
plt.ylabel('Percent Survived', fontsize =15)

#plt.show()
plt.savefig("censoring.png")
'''

cumNumberCensored=[0]

for i in range(1,60):
	cumNumberCensored.append( cumNumberCensored[i-1]+ NumberCensored[i])

plt.plot(range(60), cumNumberCensored[:60] , label = 'Cumulative Number Censored')
plt.plot(range(60), numDead[:60] , label = 'Cumulative Number of Dead Patients')
plt.plot(range(60), 1105.0 - np.array(numDead[:60]) - np.array(cumNumberCensored[:60]), label = 'Number of Patients Left in Study ')
plt.title('Population of Patients In Study Over Time',fontsize = 20 )

plt.xlabel('Months',fontsize =15)
plt.ylabel('Number of People',fontsize =15)

legend = plt.legend(loc='upper right', shadow=True, fontsize=12)


#plt.show()
plt.savefig("populations.png")



'''
print percentDead
plt.plot(numDead )
plt.xlabel('months')
plt.ylabel('%of people that die')
plt.show()



plt.plot( numCensored )
plt.xlabel('months')
plt.ylabel('percentage censored ')
plt.show()

'''



	
