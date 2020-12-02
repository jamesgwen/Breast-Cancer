#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plot 
plot.style.use('fivethirtyeight')
%matplotlib inline 

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
#from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

data_folder = '~/Downloads/Python_Projects/Cancer/Data/'
# %%

raw_data = pd.read_csv(data_folder + 'data.csv', sep=',', encoding='latin-1')
data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')] # get rid of unnamed random columns
data.drop(['id'], axis = 1, inplace = True)


# %%
data.head()
# %%
data.diagnosis.unique()
# %%
len(data)
# %%
# create a function to change diagnosis column to 0,1 binary. 1 = M. 0 = B
def diagnosis_converter(diagnosis):
    if diagnosis == 'M': 
        return 1
    elif diagnosis == 'B':
        return 0
    else: 
        return 'Error'
data['diagnosis_binary'] = data['diagnosis'].apply(diagnosis_converter)
data.diagnosis_binary.unique() # check to see function worked properly 
# %%
data.drop(['diagnosis'], axis = 1, inplace = True) # drop original diagnosis column
data = data.rename(columns={'diagnosis_binary': 'diagnosis'}) # rename new 0,1 binary column as "diagnosis"
var_names = list(data.columns[1:11])
# %%
plot.hist(data['diagnosis'])
plot.xticks([0,1])
plot.title('Diagnosis Graph', y = 1)
plot.xlabel('1 = Malignant, 0 = Benign')
plot.ylabel('Frequency')
fig = plot.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('histogram.png', dpi=100)
plot.show()

# %%
data.shape[0]

#%%
ind_var = data.drop('diagnosis', axis=1)
depen_var = data[['diagnosis']]
# %%
# from AnalyticsVidya tutorials!

def classification_model(model, data_x, data_y, predictors, outcome):

    model.fit(data[predictors], data[outcome]) # fit the model

    predictions = model.predict(data[predictors]) # get predictions from traning data

    accuracy = metrics.accuracy_score(predictions, data[outcome]) 
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    accuracy = []
    k_fold = StratifiedKFold(n_splits = 5)

    for train, test in k_fold.split(data_x,data_y):

        train_x =  data_x[predictors].iloc[train, :]

        train_y = data_y[outcome].iloc[train]

        model.fit(train_x,train_y)

        test_x = data_x[predictors].iloc[test,:]

        test_y = data_y[outcome].iloc[test]

        accuracy.append(model.score(test_x, test_y))
       

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(accuracy)))

    model.fit(train_x,train_y)
#%%
## LOGISTIC REGRESSION

predictor_var = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
outcome_var = 'diagnosis'
model = LogisticRegression()
classification_model(model,ind_var,depen_var,predictor_var,outcome_var)
#%%
## RANDOM FOREST
predictor_var = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 25, max_depth = 7, max_features = 2)
classification_model(model, ind_var, depen_var, predictor_var,outcome_var)
#%%
## RANDOM FOREST NO SIZE
predictor_var = ['texture_mean','smoothness_mean', 'compactness_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 25, max_depth = 7, max_features = 2)
classification_model(model, ind_var, depen_var, predictor_var,outcome_var)
#%%
## RANDOM FOREST Size
predictor_var = ['radius_mean', 'perimeter_mean','area_mean']
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 25, max_depth = 7, max_features = 2)
classification_model(model, ind_var, depen_var, predictor_var,outcome_var)
#%% 
## RANDOM FOREST ALL
predictor_var = var_names
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 25, max_depth = 7, max_features = 2)
classification_model(model, ind_var, depen_var, predictor_var,outcome_var)

# %%
