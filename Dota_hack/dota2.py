# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 02:15:08 2018

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


def roleconv(df_heroes):
    df_heroes['Carry'] = 2
    df_heroes['Initiator'] = 2
    df_heroes['Support'] = 2
    df_heroes['Disabler'] = 2
    df_heroes['Nuker'] = 2
    df_heroes['Escape'] = 2
    df_heroes['Durable'] = 2
    df_heroes['Pusher'] = 2
    df_heroes['Jungler'] = 2
    df_heroes['Melee'] = 2
    df_heroes['agi'] = 2
    df_heroes['str'] = 2
    df_heroes['int'] = 2
    df_heroes['Ranged']= 2
    
    for i in range(0,len(df_heroes)):
        a = df_heroes['roles'][i]
        for j in a.split(':'):
            if j in 'Carry':
                df_heroes['Carry'][i] = 1
            elif j in 'Initiator':    
                df_heroes['Initiator'][i] = 1
            elif j in 'Support':    
                df_heroes['Support'][i] = 1 
            elif j in 'Disabler':    
                df_heroes['Disabler'][i] = 1
            elif j in 'Nuker':    
                df_heroes['Nuker'][i] = 1  
            elif j in 'Escape':    
                df_heroes['Escape'][i] = 1   
            elif j in 'Durable':    
                df_heroes['Durable'][i] = 1   
            elif j in 'Pusher':    
                df_heroes['Pusher'][i] = 1   
            elif j in 'Jungler':    
                df_heroes['Jungler'][i] = 1   
            
    for i in range(0,len(df_heroes)):
        j = df_heroes['primary_attr'][i]    
        if j in 'agi':    
            df_heroes['agi'][i] = 1   
        elif j in 'str':    
            df_heroes['str'][i] = 1   
        elif j in 'int':    
            df_heroes['int'][i] = 1 
    for i in range(0,len(df_heroes)):
        j = df_heroes['attack_type'][i]
        if j in 'Melee':    
                df_heroes['Melee'][i] = 1
        elif j in 'Ranged':
             df_heroes['Ranged'][i] = 1
        
        
    lst = ['Carry', 'Initiator','Support', 'Disabler', 'Nuker', 'Escape',
           'Durable', 'Pusher','Jungler', 'Melee','Ranged', 'agi', 'str', 'int']
    for i in lst:
        df_heroes.loc[df_heroes[i]>1,i]=0
    return df_heroes    


df_hero = roleconv(hero)

#merge hero table
f9_train = train9.merge(df_hero,on='hero_id', how= 'inner')
f1_train = train1.merge(df_hero,on='hero_id', how= 'inner')

f9_test = test9.merge(df_hero,on='hero_id', how= 'inner')
f1_test = test1.merge(df_hero,on='hero_id', how= 'inner')

def gen_df(f1_train,f9_train):
    #User id wise averages
    colnames=['user_id','hero_id','Avg_games_user','Avg_kda_users']
    colnames1 = ['user_id','hero_id','Avg_games_hero','Avg_kda_hero']
    df_trainFinal=pd.DataFrame(columns=colnames)
    df_trainFinal2=pd.DataFrame(columns=colnames1)
    
    for i in f1_train['user_id'].unique():
        df_trainTemp=pd.DataFrame(columns=colnames)
        #df_trainTemp['num_games']=f1_train.loc[f1_train['user_id'] == i,'num_games']
        df_trainTemp['hero_id']= f1_train.loc[f1_train['user_id'] == i,'hero_id']
        #df_trainTemp['kda_ratio']=f1_train.loc[f1_train['user_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp['user_id']=i
        df_temp=f9_train.loc[(f9_train['user_id'] == i),:]
        df_trainTemp['Avg_games_user']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp['Avg_kda_users']=df_temp['kda_ratio'].mean()
        df_trainFinal=df_trainFinal.append(df_trainTemp)
    
    df_trainFinal=df_trainFinal.fillna(0)   
    
    #Hero id wise averages
    
    for i in f1_train['hero_id'].unique():
        df_trainTemp2=pd.DataFrame(columns=colnames1)
        df_trainTemp2['user_id']=f1_train.loc[f1_train['hero_id'] == i,'user_id']
        #df_trainTemp['num_wins']=df_train10.loc[df_train10['user_id'] == i,'num_wins']
        #df_trainTemp2['kda_ratio']=f1_train.loc[f1_train['hero_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp2['hero_id']=i
        df_temp=f9_train.loc[(f9_train['hero_id'] == i),:]
        df_trainTemp2['Avg_games_hero']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp2['Avg_kda_hero']=df_temp['kda_ratio'].mean()
        df_trainFinal2=df_trainFinal2.append(df_trainTemp2)
    
    df_trainFinal2=df_trainFinal2.fillna(0)   
    
    final_train = f1_train.merge(df_trainFinal,on='user_id')

    final_train = final_train.merge(df_trainFinal2,on='user_id')

    final_train = final_train.drop(['hero_id_x','hero_id_y'],axis=1)
    
    return final_train


final_train =  gen_df(f1_train,f9_train)

final_test =  gen_df(f1_test,f9_test)
final_test['kda_ratio'] = 1

predictor_var = ['Avg_games_user', 'Avg_kda_users', 
       'Avg_games_hero', 'Avg_kda_hero']

outcome_var = 'kda_ratio'


    



#Model, Accuracy, Imp. Variables Helper functions
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import LeaveOneOut 
loo = LeaveOneOut()
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


def model_fn(alg, X_train, Y_train, X_test, Y_test, predictor_var, degree,filename,is_poly=True,is_final=True, treemod =True):
    
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
    
    if treemod:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train[predictor_var], label=Y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=10,
            metrics='rmse', early_stopping_rounds=50,verbose_eval=True,
            show_stdv=True)
        print('best n_estimator',cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
        
        feat_imp = pd.Series(alg.booster().get_score()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
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
        df_final['id'] = final_test['id']
        df_final['kda_ratio'] = dvalid_predictions
        #df_final['kda_ratio'] = df_final['kda_ratio'].astype('str')
        df_final.to_csv(filename, index=False)
    
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
model6 = KNeighborsRegressor(n_neighbors =15)

from sklearn.linear_model import Ridge, Lasso

model7 = Ridge()

model8 = Lasso()


from vecstack import stacking

model_fn(model8,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, False)

model_f2(final_train[predictor_var],final_train[outcome_var])

#Stacking
models = [model1,model4,model5,model6,model7,model8]
    
S_train, S_test = stacking(models, final_train[predictor_var], final_train[outcome_var], final_test[predictor_var], 
    regression = True, metric = mse, n_folds = 4, 
    stratified = True, shuffle = True, random_state = 50, verbose = 2)

model1.fit(S_train,final_train[outcome_var])
y1 = model1.predict(S_test)
#print("RMSE_Score: %f" % np.sqrt(metrics.mean_squared_error(Y_test.values, y1)))



#############------------------Gsearch on Liner models------------#############

param_test1 = {
 'alpha':[1,2,3,4,5],
 'max_iter':[100, 300, 500]
}

gsearch1 = GridSearchCV(estimator = Lasso( alpha = 1,max_iter=100, random_state=50), 
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(final_train[predictor_var],final_train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


#########---------------KNN Tune-------------#############

myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, final_train[predictor_var], final_train[outcome_var], cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()    

########-----------XGB Tune-------------###########

xgb1 = XGBRegressor(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=6,
 scale_pos_weight=1,
 seed=50)

model_fn(xgb1,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, True)


param_test1 = {
 'max_depth':[5, 6, 7, 8, 9, 10,11,12,13,14,15],
 'min_child_weight':[1, 3, 5, 7]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=5,
 min_child_weight=1, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1, seed=50), 
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(final_train[predictor_var],final_train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {
 'max_depth':[4, 5, 6],
 'min_child_weight':[2,3,4]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1, seed=50), 
 param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(final_train[predictor_var],final_train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2b = {
 'min_child_weight':[4,5,6,7,8,9,10]
}
gsearch2b = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=54, max_depth=4,
 min_child_weight=2, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test2b, scoring='neg_mean_squared_error',n_jobs=6,iid=False, cv=5)
gsearch2b.fit(final_train[predictor_var],final_train[outcome_var])
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_


param_test3 = {
 'gamma':[i/10.0 for i in range(0,10)]
}
gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4,
 min_child_weight=3, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch3.fit(final_train[predictor_var],final_train[outcome_var])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

model2 = XGBRegressor(learning_rate =0.1,
 n_estimators=54,
 max_depth=4,
 min_child_weight=3,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=-1,
 scale_pos_weight=1,
 seed=50)
 
#classification_model(model, df,predictor_var,outcome_var)
model_fn(model2,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, True)

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch4.fit(final_train[predictor_var],final_train[outcome_var])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


param_test5 = {
 'subsample':[i/100.0 for i in range(75,100,5)],
 'colsample_bytree':[i/100.0 for i in range(75,100,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch5.fit(final_train[predictor_var],final_train[outcome_var])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch6.fit(final_train[predictor_var],final_train[outcome_var])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05,0.09]
}
gsearch7 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=54, max_depth=4,
 min_child_weight=3, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch7.fit(final_train[predictor_var],final_train[outcome_var])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


xgb4 = XGBRegressor(
 learning_rate =0.2,
 n_estimators=54,
 max_depth=4,
 min_child_weight=3,
 gamma=0.1,
 subsample=0.8,
 reg_alpha = 10,
 colsample_bytree=0.75,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=50)


model_fn(xgb4,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, True)

