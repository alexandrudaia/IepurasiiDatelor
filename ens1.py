
# coding: utf-8

# In[24]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('train.csv/train.csv')
test = pd.read_csv('test.csv/test.csv')
macro = pd.read_csv('macro.csv')
id_test = test.id
train.sample(3)


# In[25]:

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)        


# In[4]:

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


# In[10]:

import xgboost as xgb


# In[26]:

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)    


# In[11]:

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# In[98]:

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
prediction_list=[]
prediction_train=[]
from sklearn.model_selection import train_test_split
for i in range(30):
    print("cross val number ",i)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.3,random_state=i)
    dtrain = xgb.DMatrix(Xtrain, ytrain)
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
    
    #cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
    num_boost_rounds = len(cv_output)
    print(num_boost_rounds)
    model1 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
    y_predict1 = model1.predict(dtest)
    prediction_list.append(y_predict1)
    pred_train=model1.predict(xgb.DMatrix(x_train))
    prediction_train.append(pred_train)


# In[95]:

suma=0
for  pred in range(len(prediction_list)):
                   suma=suma+prediction_list[pred]
average=suma/len(prediction_list)

suma=0
for  pred in range(len(prediction_train)):
                   suma=suma+prediction_train[pred]
averageTrain=suma/len(prediction_train)


# In[96]:

output = pd.DataFrame({'id': id_test, 'price_doc1':average})
output.head()

output_train=pd.DataFrame({'price_doc1':averageTrain})
output.to_csv('predictionTrain1.csv',index=False)


# In[97]:

output.to_csv('predictionTest1.csv', index=False)

