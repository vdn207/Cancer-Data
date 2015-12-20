
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import exp

from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.datasets import load_waltons
from lifelines.utils import datetimes_to_durations
from lifelines.utils import survival_table_from_events
from lifelines import AalenAdditiveFitter, CoxPHFitter


from lifelines import KaplanMeierFitter

cancer = pd.read_csv("Breast Invasive Carcinoma (TCGA, Provisional).csv")

cancer = cancer.dropna(subset = ['OS MONTHS'])

cancer['T'] = cancer['OS MONTHS']
cancer['E'] = cancer['OS STATUS']=="DECEASED"


def handle_na(df):
    '''Handle the NaN fields'''

    df['AGE'].fillna(df['AGE'].mean(), inplace = True)
    df['Mutation Count'].fillna(df['Mutation Count'].median(), inplace = True)
    df['CNA'].fillna(df['CNA'].mean(), inplace = True)
    df['LYMPH NODES EXAMINED'].fillna(df['LYMPH NODES EXAMINED'].mean(), inplace = True)
    df['INITIAL WEIGHT'].fillna(df['INITIAL WEIGHT'].mean(), inplace = True)
    df['AJCC METASTASIS PATHOLOGIC PM'].fillna(df['AJCC METASTASIS PATHOLOGIC PM'].value_counts().index[0], inplace = True)
    df['AJCC NODES PATHOLOGIC PN'].fillna(df['AJCC NODES PATHOLOGIC PN'].value_counts().index[0], inplace = True)
    df['AJCC PATHOLOGIC TUMOR STAGE'].fillna(df['AJCC PATHOLOGIC TUMOR STAGE'].value_counts().index[0], inplace = True)
    df['AJCC TUMOR PATHOLOGIC PT'].fillna(df['AJCC TUMOR PATHOLOGIC PT'].value_counts().index[0], inplace = True)
    df['ETHNICITY'].fillna(df['ETHNICITY'].value_counts().index[0], inplace = True)
    df['GENDER'].fillna(df['GENDER'].value_counts().index[0], inplace = True)
    df['HISTORY NEOADJUVANT TRTYN'].fillna(df['HISTORY NEOADJUVANT TRTYN'].value_counts().index[0], inplace = True)
    df['HISTORY OTHER MALIGNANCY'].fillna(df['HISTORY OTHER MALIGNANCY'].value_counts().index[0], inplace = True)
    df['ICD 10'].fillna(df['ICD 10'].value_counts().index[0], inplace = True)
    df['ICD O 3 HISTOLOGY'].fillna(df['ICD O 3 HISTOLOGY'].value_counts().index[0], inplace = True)
    df['ICD O 3 SITE'].fillna(df['ICD O 3 SITE'].value_counts().index[0], inplace = True)
    
    return df 

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

features_chosen_by_domain_knowledge = ['AGE', 'AJCC METASTASIS PATHOLOGIC PM', 'AJCC NODES PATHOLOGIC PN', 'AJCC PATHOLOGIC TUMOR STAGE', 'AJCC TUMOR PATHOLOGIC PT', 'ETHNICITY', 'GENDER', 'CNA', 'INITIAL WEIGHT', 'HISTORY NEOADJUVANT TRTYN', 'HISTORY OTHER MALIGNANCY', 'LYMPH NODES EXAMINED', 'Mutation Count','ICD 10', 'ICD O 3 HISTOLOGY', 'ICD O 3 SITE','T','E']

cancer= cancer[features_chosen_by_domain_knowledge]

catVar = ['AJCC METASTASIS PATHOLOGIC PM', 'AJCC NODES PATHOLOGIC PN', 'AJCC PATHOLOGIC TUMOR STAGE', 'AJCC TUMOR PATHOLOGIC PT', 'ETHNICITY', 'GENDER', 'HISTORY NEOADJUVANT TRTYN', 'HISTORY OTHER MALIGNANCY', 'ICD 10', 'ICD O 3 HISTOLOGY', 'ICD O 3 SITE']
cancer = handle_na(cancer)
cancer = handle_categorical_variables(cancer,catVar )



#cancer=cancer.drop(['OS MONTHS'], axis = 1 )
#cancer=cancer.drop(['OS STATUS'], axis =1 )

'''
`
kmf = KaplanMeierFitter()



#make table
table = survival_table_from_events(T, E)
print table.head()

#make kaplanMeier curve
kmf.fit(durations, event_observed=events) # more succiently, kmf.fit(T,E)

kmf.survival_function_
kmf.median_
kmf.plot()
plt.show()


#cool graphs for two groups
groups = cancer['RACE']
ix = (groups == 'WHITE')

kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='WHITE')
kmf.plot(ax=ax)
plt.show()



'''

#print cancer['T'].unique()
#print cancer['E'].unique()
#cancer = cancer.dropna()


# the '-1' term
# refers to not adding an intercept column (a column of all 1s).
# It can be added to the Fitter class.

covMatrix = cancer.cov()

cf = CoxPHFitter()
cf.fit(covMatrix, 'T', event_col= 'E')  #extra paramater for categorical , strata=catVar
cf.print_summary()

curve = cf.predict_survival_function(cancer)
curve.plot()
plt.show()
print "hazard coeff",cf.hazards_
print "baseline ", cf.baseline_hazard_

'''
scores = k_fold_cross_validation(cf, covMatrix, 'T', event_col='E', k=3)
print scores
print np.mean(scores)
print np.std(scores)

'''

