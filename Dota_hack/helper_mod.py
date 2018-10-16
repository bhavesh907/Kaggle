# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:05:58 2018

@author: uesr
"""

#import libraries
import pandas as pd
import numpy as np
#from ggplot import *
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

#import raw data
df_train9=pd.read_csv("C:/Users/uesr/Downloads/Dota_hack/train9.csv")
df_train10=pd.read_csv("C:/Users/uesr/Downloads/Dota_hack/train1.csv")
df_test9=pd.read_csv("C:/Users/uesr/Downloads/Dota_hack/test9.csv")
df_test10=pd.read_csv("C:/Users/uesr/Downloads/Dota_hack/test1.csv")
df_heroes=pd.read_csv("C:/Users/uesr/Downloads/Dota_hack/hero_data.csv")

sample = pd.read_csv('C:/Users/uesr/Downloads/Dota_hack/sample_submission_CKEH6IJ.csv')

#making deviation=1 in every variable for kmean clustering
for i in ['base_health_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence', 'strength_gain',
       'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed',
       'attack_rate', 'move_speed', 'turn_rate']:
    df_heroes[i]=df_heroes[i]/df_heroes[i].std()

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

#array for kmean clustering
array=df_heroes.loc[:,[
       'base_health_regen', 'base_armor',
       'base_magic_resistance', 'base_attack_min', 'base_attack_max',
       'base_strength', 'base_agility', 'base_intelligence', 'strength_gain',
       'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed',
       'attack_rate', 'move_speed', 'turn_rate', 'Carry', 'Initiator',
       'Support', 'Disabler', 'Nuker', 'Escape', 'Durable', 'Pusher',
       'Jungler', 'Melee','Ranged', 'agi', 'str', 'int']]

# elbow curve
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(array)
    kmeanModel.fit(array)
    distortions.append(sum(np.min(cdist(array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / array.shape[0])
 
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#kmean clustering and adding labels
kmeanModel = KMeans(n_clusters=3).fit(array)
df_heroes['cluster']=kmeanModel.labels_

#adding heroes_cluster to every data frame
df_herolab=df_heroes.loc[:,['hero_id','cluster']]
df_train9=df_train9.merge(df_herolab,left_on='hero_id',right_on='hero_id', how='left' )
df_train10=df_train10.merge(df_herolab,left_on='hero_id',right_on='hero_id', how='left' )
df_test9=df_test9.merge(df_herolab,left_on='hero_id',right_on='hero_id', how='left' )
df_test10=df_test10.merge(df_herolab,left_on='hero_id',right_on='hero_id', how='left' )


            
    