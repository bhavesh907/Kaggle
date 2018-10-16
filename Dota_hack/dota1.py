# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:08:35 2018

@author: uesr
"""

#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from vecstack import stacking

hero = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/hero_data.csv')
sample = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/sample_submission_CKEH6IJ.csv')
#final test data
test1 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/test1.csv')
#validation set for hyper parameter tuning
test9 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/test9.csv')
#training-testing sets
train1 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/train1.csv')
train9 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/train9.csv')

#merge hero table
f9_train = train9.merge(hero,on='hero_id', how= 'inner')
f1_train = train1.merge(hero,on='hero_id', how= 'inner')

f9_test = test9.merge(hero,on='hero_id', how= 'inner')
f1_test = test1.merge(hero,on='hero_id', how= 'inner')


predictor_var = ['num_games', 'num_wins', 
       'primary_attr', 'attack_type', 'roles', 'base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence', 'strength_gain',
       'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed',
       'attack_rate', 'move_speed', 'turn_rate']


outcome_var = 'kda_ratio'

labencode = ['roles']

hotencode = ['primary_attr','attack_type']

le = LabelEncoder()

def encode(df, labencode,hotencode,predictor_var,outcome_var):
    for i in labencode:
        df[i] = le.fit_transform(df[i])
    df.dtypes
    
    X1= pd.get_dummies(df[hotencode])
    X,y  = df[predictor_var].drop(hotencode,axis=1), df[outcome_var]
    X = pd.concat([X,X1], axis=1)
    return X,y

def encode_test(df, labencode,hotencode,predictor_var):
    for i in labencode:
        df[i] = le.fit_transform(df[i])
    df.dtypes
    
    X1= pd.get_dummies(df[hotencode])
    X  = df[predictor_var].drop(hotencode,axis=1)
    X = pd.concat([X,X1], axis=1)
    return X

#Final datasets
X_train,Y_train = encode(f9_train,labencode,hotencode,predictor_var,outcome_var)
X_test,Y_test = encode(f1_train,labencode,hotencode,predictor_var,outcome_var)

X_valid,Y_valid = encode(f9_test,labencode,hotencode,predictor_var,outcome_var)

predictor_var1 = ['num_games', 
       'primary_attr', 'attack_type', 'roles', 'base_health',
       'base_health_regen', 'base_mana', 'base_mana_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence', 'strength_gain',
       'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed',
       'attack_rate', 'move_speed', 'turn_rate']

X_final_test = encode_test(f1_test,labencode,hotencode,predictor_var1)

#Model, Accuracy, Imp. Variables Helper functions
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import LeaveOneOut 
loo = LeaveOneOut()
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

def model_fn(alg, X_train, Y_train, X_test, Y_test, predictor_var, degree,filename,is_poly=True,is_final=True):
    
    #Kfold cross validation
    
    #kf = loo.split(X_train_std[predictor_var])
    kf = KFold(n_splits=5).split(X_train[predictor_var])
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (X_train[predictor_var].iloc[train,:])
        
        # The target we're using to train the algorithm.
        train_target = Y_train.iloc[train]
        
        # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
        y1 = alg.predict(X_train[predictor_var].iloc[test,:])
        
        #Record error from each cross-validation run
        error.append(np.sqrt(metrics.mean_squared_error(Y_train.iloc[test], y1)))
     
    print ("Cross-Validation RMSE Score : %f" % np.mean(error))
    
    if is_poly:
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = poly_reg.fit_transform(X_train[predictor_var])
        poly_reg.fit(X_poly, Y_train)
        alg.fit(X_poly, Y_train)
    
    else:
        #Fit the algorithm on the data
        alg.fit(X_train[predictor_var], Y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train[predictor_var])
    
    dvalid_predictions = alg.predict(X_test[predictor_var])
    
    if is_final:
        #Print model report:
        print("Writing Final Csv")
        print("\nModel Report")
        print("RMSE_Score_(Train): %f" % np.sqrt(metrics.mean_squared_error(Y_train.values, dtrain_predictions)))
        #print("R2_Score_(Train): %f" % alg.score(X_train[predictor_var],Y_train))
        print("R2_Score_(Train): %f" % metrics.r2_score(Y_train,dtrain_predictions))
        
        df_final = pd.DataFrame()
        df_final['kda_ratio'] = dvalid_predictions
        df_final['id'] = test1['id']
        #df_final['kda_ratio'] = df_final['kda_ratio'].astype('str')
        df_final.to_csv(filename, index=True)
    
    else:
        #Print model report:
        print("\nModel Report")
        print("RMSE_Score_(Train): %f" % np.sqrt(metrics.mean_squared_error(Y_train.values, dtrain_predictions)))
        #print("R2_Score_(Train): %f" % alg.score(X_train[predictor_var],Y_train))
        print("R2_Score_(Train): %f" % metrics.r2_score(Y_train,dtrain_predictions))
    
        print("RMSE Score (Test): %f" % np.sqrt(metrics.mean_squared_error(Y_test.values, dvalid_predictions)))
        #print("R2 Score (Test): %f" % alg.score(X_test[predictor_var],Y_test))
        print("R2_Score_(Test): %f" % metrics.r2_score(Y_test,dvalid_predictions))
    

import statsmodels.api as sm

def model_f2(X_train,Y_train):
    mod = sm.OLS(Y_train, X_train)
    res = mod.fit()
    print(res.summary())

#Correlation matrix    
'''x1= X_train.drop(['num_wins','base_attack_min','attack_type_Ranged','primary_attr_int','move_speed'],axis=1)
corr_df = X_train.corr(method = 'pearson')
mask = np.zeros_like(corr_df)
mask[np.triu_indices_from(mask)] = True

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))
sns.regplot(x="num_games", y="roles", data=X_train);
f, ax = plt.subplots(figsize=(5, 6))
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

sns.jointplot(x="num_wins", y="num_games", data=X_train, kind="reg");

sns.heatmap(corr_df,cmap='YlGnBu',vmax=1.0,vmin=-1.0,mask=mask,linewidths=1,ax=ax)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.show()

model_f2(x1,Y_train)
'''


#plot_df = pd.concat([X_train,Y_train],axis =1)
#sns.pairplot(plot_df)

#Generate models
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

'''
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, Y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y_train)'''

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
model2 = SVR(kernel = 'rbf')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
model3 = DecisionTreeRegressor(random_state = 50)

from sklearn.ensemble import RandomForestRegressor
model4 = RandomForestRegressor(n_estimators = 300, random_state = 50)

from xgboost import XGBRegressor
model5 = XGBRegressor()

from sklearn.neighbors import KNeighborsRegressor
model6 = KNeighborsRegressor()


predictor_var = ['num_games', 'roles', 'base_health', 'base_health_regen',
       'base_mana', 'base_mana_regen', 'base_armor', 'base_magic_resistance',
       'base_attack_min', 'base_attack_max', 'base_strength', 'base_agility',
       'base_intelligence', 'strength_gain', 'agility_gain',
       'intelligence_gain', 'attack_range', 'projectile_speed', 'attack_rate',
       'move_speed', 'turn_rate', 'primary_attr_agi', 'primary_attr_int',
       'primary_attr_str', 'attack_type_Melee', 'attack_type_Ranged']

from vecstack import stacking

model_fn(model5,X_train,Y_train,X_final_test,Y_valid,predictor_var, 3,'alg2.csv', False, False)


#Stacking
models = [model1,model3,model4,model5,model6]
    
S_train, S_test = stacking(models, X_train[predictor_var], Y_train, X_test[predictor_var], 
    regression = True, metric = mse, n_folds = 4, 
    stratified = True, shuffle = True, random_state = 50, verbose = 2)

model1.fit(S_train,Y_train)
y1 = model1.predict(S_test)
print("RMSE_Score: %f" % np.sqrt(metrics.mean_squared_error(Y_test.values, y1)))




