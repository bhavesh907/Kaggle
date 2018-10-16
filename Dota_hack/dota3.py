# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:04:27 2018

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

#import raw data
hero = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/hero_data.csv')
sample = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/sample_submission_CKEH6IJ.csv')
#final test data
test1 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/test1.csv')
#validation set for hyper parameter tuning
test9 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/test9.csv')
#training-testing sets
train1 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/train1.csv')
train9 = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/train9.csv')

#import raw data with clusters-implemented in helper mod

train1 = df_train10
train9 = df_train9
test1 = df_test10
test9 = df_test9
df_hero = df_heroes

############################################################################
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
f9_train = f9_train.drop(['cluster_y'],axis=1)


f1_train = train1.merge(df_hero,on='hero_id', how= 'inner')
f1_train = f1_train.drop(['cluster_y'],axis=1)

f9_test = test9.merge(df_hero,on='hero_id', how= 'inner')
f9_test = f9_test.drop(['cluster_y'],axis=1)

f1_test = test1.merge(df_hero,on='hero_id', how= 'inner')
f1_test = f1_test.drop(['cluster_y'],axis=1)

def gen_df(f1_train,f9_train):
    #User id wise averages
    colnames=['user_id','hero_id','Avg_games_user','Avg_kda_users']
    colnames1 = ['user_id','hero_id','Avg_games_hero','Avg_kda_hero']
    colnames2 = ['user_id','hero_id','Avg_Carrygames_user','Avg_Carrykda_user',
                 'Avg_Inigames_user','Avg_Inikda_user','Avg_Suppgames_user','Avg_Suppkda_user',
                 'Avg_Dsbgames_user','Avg_Dsbkda_user','Avg_Nukgames_user','Avg_Nukkda_user','Avg_Escgames_user',
                 'Avg_Esckda_user','Avg_Durgames_user','Avg_Durkda_user','Avg_Pusgames_user',
                 'Avg_Puskda_user','Avg_Jungames_user','Avg_Junkda_user']
    colnames3 = ['user_id','hero_id','Avg_C0_games_user','Avg_C0_kda_user',
                 'Avg_C1_games_user','Avg_C1_kda_user','Avg_C2_games_user','Avg_C2_kda_user']
    colnames4 = ['user_id','hero_id','Avg_Melgames_user','Avg_Melkda_user','Avg_Rangames_user','Avg_Rankda_user']
    
    colnames5 = ['user_id','hero_id','Avg_Carrygames_hero','Avg_Carrykda_hero','Avg_Inigames_hero','Avg_Inikda_hero',
                 'Avg_Suppgames_hero','Avg_Suppkda_hero','Avg_Dsbgames_hero','Avg_Dsbkda_hero','Avg_Nukgames_hero','Avg_Nukkda_hero',
                 'Avg_Escgames_hero','Avg_Esckda_hero','Avg_Durgames_hero','Avg_Durkda_hero',
                 'Avg_Pusgames_hero','Avg_Puskda_hero','Avg_Jungames_hero','Avg_Junkda_hero','Avg_C0_games_hero','Avg_C0_kda_hero',
                 'Avg_C1_games_hero','Avg_C1_kda_hero','Avg_C2_games_hero','Avg_C2_kda_hero',
                 'Avg_Melgames_hero','Avg_Melkda_hero','Avg_Rangames_hero','Avg_Rankda_hero']
    
    df_trainFinal=pd.DataFrame(columns=colnames)
    df_trainFinal2=pd.DataFrame(columns=colnames1)
    df_trainFinal3=pd.DataFrame(columns=colnames2)
    df_trainFinal4=pd.DataFrame(columns=colnames3)
    df_trainFinal5=pd.DataFrame(columns=colnames4)
    df_trainFinal6=pd.DataFrame(columns=colnames5)
    
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
    
    # Role wise avgs
    
    for i in f1_train['user_id'].unique():
        df_trainTemp3=pd.DataFrame(columns=colnames2)
        #df_trainTemp['num_games']=f1_train.loc[f1_train['user_id'] == i,'num_games']
        df_trainTemp3['hero_id']= f1_train.loc[f1_train['user_id'] == i,'hero_id']
        #df_trainTemp['kda_ratio']=f1_train.loc[f1_train['user_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp3['user_id']=i
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Carry']==1),:]
        df_trainTemp3['Avg_Carrygames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Carrykda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Initiator']==1),:]
        df_trainTemp3['Avg_Inigames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Inikda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Support']==1),:]
        df_trainTemp3['Avg_Suppgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Suppkda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Disabler']==1),:]
        df_trainTemp3['Avg_Dsbgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Dsbkda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Nuker']==1),:]
        df_trainTemp3['Avg_Nukgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Nukkda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Escape']==1),:]
        df_trainTemp3['Avg_Escgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Esckda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Durable']==1),:]
        df_trainTemp3['Avg_Durgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Durkda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Pusher']==1),:]
        df_trainTemp3['Avg_Pusgames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Puskda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Jungler']==1),:]
        df_trainTemp3['Avg_Jungames_user']=df_temp['num_games'].mean()
        df_trainTemp3['Avg_Junkda_user']=df_temp['kda_ratio'].mean()

        df_trainFinal3=df_trainFinal3.append(df_trainTemp3)
    
    df_trainFinal3=df_trainFinal3.fillna(0)   
    
    # cluster wise avgs
    
    for i in f1_train['user_id'].unique():
        df_trainTemp4=pd.DataFrame(columns=colnames3)
        #df_trainTemp['num_games']=f1_train.loc[f1_train['user_id'] == i,'num_games']
        df_trainTemp4['hero_id']= f1_train.loc[f1_train['user_id'] == i,'hero_id']
        #df_trainTemp['kda_ratio']=f1_train.loc[f1_train['user_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp4['user_id']=i
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['cluster_x']==0),:]
        df_trainTemp4['Avg_C0_games_user']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp4['Avg_C0_kda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['cluster_x']==1),:]
        df_trainTemp4['Avg_C1_games_user']=df_temp['num_games'].mean()
        df_trainTemp4['Avg_C1_kda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['cluster_x']==2),:]
        df_trainTemp4['Avg_C2_games_user']=df_temp['num_games'].mean()
        df_trainTemp4['Avg_C2_kda_user']=df_temp['kda_ratio'].mean()
        
        df_trainFinal4=df_trainFinal4.append(df_trainTemp4)
    
    df_trainFinal4=df_trainFinal4.fillna(0)   
    
    #Attack wise avgs
    
    for i in f1_train['user_id'].unique():
        df_trainTemp5=pd.DataFrame(columns=colnames4)
        #df_trainTemp['num_games']=f1_train.loc[f1_train['user_id'] == i,'num_games']
        df_trainTemp5['hero_id']= f1_train.loc[f1_train['user_id'] == i,'hero_id']
        #df_trainTemp['kda_ratio']=f1_train.loc[f1_train['user_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp5['user_id']=i
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Melee']==1),:]
        df_trainTemp5['Avg_Melgames_user']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp5['Avg_Melkda_user']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['user_id'] == i) & (f9_train['Ranged']==1),:]
        df_trainTemp5['Avg_Rangames_user']=df_temp['num_games'].mean()
        df_trainTemp5['Avg_Rankda_user']=df_temp['kda_ratio'].mean()
        
        df_trainFinal5=df_trainFinal5.append(df_trainTemp5)
    
    df_trainFinal5=df_trainFinal5.fillna(0)
    
    #Hero id wise all averages
    
    for i in f1_train['hero_id'].unique():
        df_trainTemp6=pd.DataFrame(columns=colnames5)
        df_trainTemp6['user_id']=f1_train.loc[f1_train['hero_id'] == i,'user_id']
        #df_trainTemp['num_wins']=df_train10.loc[df_train10['user_id'] == i,'num_wins']
        #df_trainTemp2['kda_ratio']=f1_train.loc[f1_train['hero_id'] == i,'kda_ratio']
        #df_trainTemp['cluster']=df_train10.loc[df_train10['user_id'] == i,'cluster']
        df_trainTemp6['hero_id']=i
        
        ##################---------Role avgs----------#########################
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Carry']==1),:]
        df_trainTemp6['Avg_Carrygames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Carrykda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Initiator']==1),:]
        df_trainTemp6['Avg_Inigames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Inikda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Support']==1),:]
        df_trainTemp6['Avg_Suppgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Suppkda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Disabler']==1),:]
        df_trainTemp6['Avg_Dsbgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Dsbkda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Nuker']==1),:]
        df_trainTemp6['Avg_Nukgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Nukkda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Escape']==1),:]
        df_trainTemp6['Avg_Escgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Esckda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Durable']==1),:]
        df_trainTemp6['Avg_Durgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Durkda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Pusher']==1),:]
        df_trainTemp6['Avg_Pusgames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Puskda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Jungler']==1),:]
        df_trainTemp6['Avg_Jungames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Junkda_hero']=df_temp['kda_ratio'].mean()
        
        ###############-----------cluster avgs-----------####################
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['cluster_x']==0),:]
        df_trainTemp6['Avg_C0_games_hero']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp6['Avg_C0_kda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['cluster_x']==1),:]
        df_trainTemp6['Avg_C1_games_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_C1_kda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['cluster_x']==2),:]
        df_trainTemp6['Avg_C2_games_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_C2_kda_hero']=df_temp['kda_ratio'].mean()
        
        #######################-----------attack avgs------------------################
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Melee']==1),:]
        df_trainTemp6['Avg_Melgames_hero']=df_temp['num_games'].mean()
        #df_trainTemp['0_num_wins']=df_temp['num_wins'].sum()
        df_trainTemp6['Avg_Melkda_hero']=df_temp['kda_ratio'].mean()
        
        df_temp=f9_train.loc[(f9_train['hero_id'] == i) & (f9_train['Ranged']==1),:]
        df_trainTemp6['Avg_Rangames_hero']=df_temp['num_games'].mean()
        df_trainTemp6['Avg_Rankda_hero']=df_temp['kda_ratio'].mean()
        
        df_trainFinal6=df_trainFinal6.append(df_trainTemp6)
        
    df_trainTemp6['Avg_Rankda_hero'].fillna(df_trainTemp6['Avg_Rankda_hero'].mean(), inplace=True)

    df_trainFinal6=df_trainFinal6.fillna(0)   
    
    
    final_train = f1_train.merge(df_trainFinal,on='user_id')

    final_train = final_train.merge(df_trainFinal2,on='user_id')
    
    final_train = final_train.merge(df_trainFinal3,on='user_id')
    
    final_train = final_train.merge(df_trainFinal4,on='user_id')
    
    final_train = final_train.merge(df_trainFinal5,on='user_id')
    
    final_train = final_train.merge(df_trainFinal6,on='user_id')

    final_train = final_train.drop(['hero_id_x','hero_id_y'],axis=1)
    
    return final_train


final_train =  gen_df(f1_train,f9_train)

final_test =  gen_df(f1_test,f9_test)


final_test['kda_ratio'] = 1

predictor_var = [
 'base_health',
 'base_health_regen',
 'base_mana',
 'base_mana_regen',
 'base_armor',
 'base_attack_min',
 'base_attack_max',
 'strength_gain',
 'turn_rate',
 'Nuker',
 'agi',
 'Avg_games_user',
 'Avg_kda_users',
 'Avg_kda_hero',
 'Avg_Carrykda_user',
 'Avg_Inikda_user',
  'Avg_Suppkda_user',
 'Avg_Dsbkda_user',
 'Avg_Nukgames_user',
 'Avg_Esckda_user',
 'Avg_C0_kda_user',
 'Avg_C1_kda_user',
 'Avg_Suppgames_hero',
 'Avg_Nukgames_hero',
 'Avg_Esckda_hero',
 'Avg_Pusgames_hero',
 'Avg_Junkda_hero',
 'Avg_Melkda_hero']
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
    kf = KFold(n_splits=10).split(X_train[predictor_var])
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
        print(feat_imp.transpose)
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
        
        
#################-----------------------------Scaled Feature model function------------------------------##############
def model_fn_scaled(alg, X_train, Y_train, X_test, Y_test, degree,filename,is_poly=True,is_final=True, treemod =True):
    
    #Kfold cross validation
    
    #kf = loo.split(X_train_std[predictor_var])
    kf = KFold(n_splits=10).split(X_train)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (X_train[train,:])
        
        # The target we're using to train the algorithm.
        train_target = Y_train[train]
        
        # Training the algorithm using the predictors and target.
        alg.fit(train_predictors, train_target)
        y1 = alg.predict(X_train[test,:])
        
        #Record error from each cross-validation run
        error.append(np.sqrt(metrics.mean_squared_error(Y_train[test], y1)))
     
    print ("Cross-Validation RMSE Score : %f" % np.mean(error))
    
    if treemod:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=Y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=10,
            metrics='rmse', early_stopping_rounds=50,verbose_eval=True,
            show_stdv=True)
        print('best n_estimator',cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
        
        feat_imp = pd.Series(alg.booster().get_score()).sort_values(ascending=False)
        print(feat_imp.transpose)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    
    if is_poly:
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = poly_reg.fit_transform(X_train)
        poly_reg.fit(X_poly, Y_train)
        alg.fit(X_poly, Y_train)
    
    else:
        #Fit the algorithm on the data
        alg.fit(X_train, Y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    
    dvalid_predictions = alg.predict(X_test)
    
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
model4 = RandomForestRegressor(n_estimators = 270, max_depth=5,
                               min_samples_split=30,min_samples_leaf=12,max_leaf_nodes=30,min_weight_fraction_leaf=0.1,random_state = 50)

from xgboost import XGBRegressor
model5 = XGBRegressor()

from sklearn.neighbors import KNeighborsRegressor
model6 = KNeighborsRegressor(n_neighbors =37)

from sklearn.linear_model import Ridge, Lasso ,ElasticNet

model7 = Ridge()

model8 = Lasso(max_iter=1000,alpha=2000)

model9 = RandomForestRegressor()

model10 = ElasticNet()

from vecstack import stacking

model_fn(model10,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, False)

model_f2(final_train[predictor_var],final_train[outcome_var])

#Stacking
models = [model1,model3,model4,model5,model6,model7,model9,model10]
    
S_train, S_test = stacking(models, final_train[predictor_var], final_train[outcome_var], final_test[predictor_var], 
    regression = True, metric = mse, n_folds = 10, 
    stratified = False, shuffle = True, random_state = 50, verbose = 2)

models = [model4,model5,model7,model9]

S_train1, S_test1 = stacking(models,S_train, final_train[outcome_var], S_test, 
    regression = True, metric = mse, n_folds = 10, 
    stratified = False, shuffle = True, random_state = 50, verbose = 2)

models = [model8]

S_train2, S_test2 = stacking(models,S_train1, final_train[outcome_var], S_test1, 
    regression = True, metric = mse, n_folds = 10, 
    stratified = False, shuffle = True, random_state = 50, verbose = 2)


model1.fit(S_train2,final_train[outcome_var])
y1 = model1.predict(S_test2)
#print("RMSE_Score: %f" % np.sqrt(metrics.mean_squared_error(Y_test.values, y1)))
df_final = pd.DataFrame()       
df_final['id'] = final_test['id']
df_final['kda_ratio'] = y1
    #df_final['kda_ratio'] = df_final['kda_ratio'].astype('str')
df_final.to_csv('Stacking.csv', index=False)

##############----------------Scaled models-----------------------##############

from sklearn.preprocessing import StandardScaler
slc= StandardScaler()
X_train_std = slc.fit_transform(final_train[predictor_var])
X_test_std = slc.fit_transform(final_test[predictor_var])

model_fn_scaled(model4,X_train_std,final_train[outcome_var],
         X_test_std,final_test[outcome_var],
         3,'alg2.csv', False, True, False)





#############------------------Gsearch on Liner models------------#############

param_test1 = {
 'alpha':[0,1,2,3,4,5,10,50,100,200,300,500,1000,2000,3000,4000,5000],
 'max_iter':[100, 300, 500,1000,3000,5000]
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
    scores = cross_val_score(knn, final_train[predictor_var], final_train[outcome_var], cv=10, scoring='neg_mean_squared_error')
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
 'max_depth':[1,2,3,4,5, 6, 7, 8, 9, 10,11,12,13,14,15],
 'min_child_weight':[1, 3, 5, 7]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=5,
 min_child_weight=1, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1, seed=50), 
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch1.fit(final_train[predictor_var],final_train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {
 'max_depth':[1, 2, 3],
 'min_child_weight':[6,7,8]
}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1, seed=50), 
 param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch1.fit(final_train[predictor_var],final_train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2b = {
 'min_child_weight':[3,4,5,6,7,8,9,10]
}
gsearch2b = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=82, max_depth=2,
 min_child_weight=2, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test2b, scoring='neg_mean_squared_error',n_jobs=6,iid=False, cv=10)
gsearch2b.fit(final_train[predictor_var],final_train[outcome_var])
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_


param_test3 = {
 'gamma':[i/10.0 for i in range(0,10)]
}
gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=2,
 min_child_weight=10, gamma=0,subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch3.fit(final_train[predictor_var],final_train[outcome_var])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

model2 = XGBRegressor(learning_rate =0.1,
 n_estimators=82,
 max_depth=2,
 min_child_weight=10,
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

gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=2,
 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch4.fit(final_train[predictor_var],final_train[outcome_var])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


param_test5 = {
 'subsample':[i/100.0 for i in range(85,100,5)],
 'colsample_bytree':[i/100.0 for i in range(55,70,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=2,
 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch5.fit(final_train[predictor_var],final_train[outcome_var])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=2,
 min_child_weight=10, gamma=0, subsample=0.9, colsample_bytree=0.6,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch6.fit(final_train[predictor_var],final_train[outcome_var])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


param_test7 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05,0.09,100,120,150,200]
}
gsearch7 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=82, max_depth=2,
 min_child_weight=10, gamma=0,subsample=0.9, colsample_bytree=0.6,
 objective= 'reg:linear', nthread=-1, scale_pos_weight=1,seed=50), 
 param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=10)
gsearch7.fit(final_train[predictor_var],final_train[outcome_var])
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


xgb4 = XGBRegressor(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=2,
 min_child_weight=10,
 gamma=0,
 subsample=0.9,
 reg_alpha = 120,
 colsample_bytree=0.6,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=50)


model_fn(xgb4,final_train[predictor_var],final_train[outcome_var],
         final_test[predictor_var],final_test[outcome_var],predictor_var,
         3,'alg2.csv', False, True, True)

