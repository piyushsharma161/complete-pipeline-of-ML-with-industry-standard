# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:14:30 2020

@author: OF65
"""

import numpy as np
import pandas as pd
import joblib
from application_logging.logger import App_Logger
from datatransformation.transformData import dataTransform
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.ensemble import BalancedBaggingClassifier , BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import Pipeline

class Model_Finder:

     """
               Tthis is to find the best model

               """

     def __init__(self):  
         self.file_object = open("../logs/modeltune/log.txt", 'a+')
         self.saved_best_model_path = '../saved_model/best_model.sav'
         self.logger = App_Logger()
         self.transformed_data = dataTransform()
         self.df = self.transformed_data.trainingData()
         self.data = self.df.iloc[:,:-1]
         self.label= self.df.iloc[:,-1]
         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, 
                                 self.label, test_size = 0.2, random_state = 0,stratify=self.label)
         self.BRF = BalancedRandomForestClassifier(n_jobs=-1)
         self.EEC = EasyEnsembleClassifier(n_jobs=-1)
     def f2_make(self, y_true, y_pred):
         return fbeta_score(y_true, y_pred, beta=2)
     def get_best_params_for_balanced_random_forest(self,X_train,y_train):
         self.logger.log(self.file_object, 
              'Entered the get_best_params_for_balanced_random_forest method of the Model_Finder class')
         #def f2_make(y_true, y_pred):
             #return fbeta_score(y_true, y_pred, beta=2)
             
         print('in RF')
         f2 = make_scorer(self.f2_make)
         try:
             # Number of trees in random forest
             n_estimators = [80,100,130,160]
             criterion = ['gini', 'entropy']
             # Number of features to consider at every split
             max_features = ['log2', 'sqrt']
             # Maximum number of levels in tree
             max_depth = [5,8,10,15]
             max_depth.append(None)
             # Minimum number of samples required to split a node
             min_samples_split = [2, 5, 8]
             # Minimum number of samples required at each leaf node
             min_samples_leaf = [2, 4]
             # Method of selecting samples for training each tree
             bootstrap = [True, False]
             replacement = [True, False]
             class_weight = ['balanced', None]


             # Create the random grid
             self.param_grid = {'brf__n_estimators': n_estimators,
                                'brf__criterion' : criterion,
                                'brf__max_features': max_features,
                                'brf__max_depth': max_depth,
                                'brf__min_samples_split': min_samples_split,
                                'brf__min_samples_leaf': min_samples_leaf,
                                'brf__bootstrap': bootstrap,
                                'brf__replacement' : replacement,
                                'brf__class_weight' : class_weight}
             self.estimators = []
             #estimators.append(('standardize', StandardScaler()))
             self.estimators.append(('brf', self.BRF))
             self.pipeline_imlearn = Pipeline(self.estimators)
             self.brf_random = RandomizedSearchCV(estimator = self.pipeline_imlearn, param_distributions = self.param_grid, n_iter = 80, cv = 5, 
                               verbose=0, random_state=42, scoring = f2, n_jobs = -1)
             self.brf_random.fit(X_train,y_train)
             self.n_estimators = self.brf_random.best_params_['brf__n_estimators']
             self.criterion = self.brf_random.best_params_['brf__criterion']
             self.max_features = self.brf_random.best_params_['brf__max_features']
             self.max_depth = self.brf_random.best_params_['brf__max_depth']
             self.min_samples_split = self.brf_random.best_params_['brf__min_samples_split']
             self.min_samples_leaf = self.brf_random.best_params_['brf__min_samples_leaf']
             self.bootstrap = self.brf_random.best_params_['brf__bootstrap']
             self.replacement = self.brf_random.best_params_['brf__replacement']
             self.class_weight = self.brf_random.best_params_['brf__class_weight']
             
             self.brf = BalancedRandomForestClassifier(n_estimators = self.n_estimators, 
                                                       criterion = self.criterion,
                                                       max_features=self.max_features,
                                                       max_depth=self.max_depth,
                                                       min_samples_split=self.min_samples_split,
                                                       min_samples_leaf=self.min_samples_leaf,
                                                       bootstrap=self.bootstrap,
                                                       replacement=self.replacement,
                                                       class_weight=self.class_weight
                                                       )
             self.brf.fit(X_train,y_train)
             self.logger.log(self.file_object,
                                   'Balanced Random Forest best params: '+str(self.brf_random.best_params_)+'\t' + str(self.brf_random.best_score_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')
             print('RF done and exited')
             return self.brf
         except Exception as e:
              self.logger.log(self.file_object,
                                   'Exception occured in get_best_params_for_balanced_random_forest method of the Model_Finder class. Exception message:  ' + str(e))
              self.logger.log(self.file_object,
                                   'Balance Random Forest Parameter tuning  failed. Exited the get_best_params_for_balanced_random_forest method of the Model_Finder class')
              raise Exception()
              
     def get_best_params_for_balanced_adaBoost(self,X_train,y_train):
         self.logger.log(self.file_object, 
              'Entered the get_best_params_for_balanced_adaBoost method of the Model_Finder class')

             
         print('enter ada boost')
         f2 = make_scorer(self.f2_make)
         try:
             n_estimators = [10,15,20,25]
             warm_start = [True, False]
             sampling_strategy = ['auto', 'majority']
             replacement = [True, False]

             # Create the random grid
             self.param_grid = {'eec__n_estimators': n_estimators,
                                'eec__warm_start' : warm_start,
                                'eec__sampling_strategy': sampling_strategy,
                                'eec__replacement' : replacement}

             self.estimators = []
             #estimators.append(('standardize', StandardScaler()))
             self.estimators.append(('eec', self.EEC))
             self.pipeline_imlearn = Pipeline(self.estimators)
             self.eec_random = RandomizedSearchCV(estimator = self.pipeline_imlearn, param_distributions = self.param_grid, n_iter = 32, cv = 5, 
                               verbose=0, random_state=42, scoring = f2, n_jobs = -1)
             self.eec_random.fit(X_train,y_train)
             self.n_estimators = self.eec_random.best_params_['eec__n_estimators']
             self.warm_start = self.eec_random.best_params_['eec__warm_start']
             self.sampling_strategy = self.eec_random.best_params_['eec__sampling_strategy']
             self.replacement = self.eec_random.best_params_['eec__replacement']

             
             self.eec = EasyEnsembleClassifier(n_estimators = self.n_estimators, 
                                                       warm_start = self.warm_start,
                                                       sampling_strategy=self.sampling_strategy,
                                                       replacement=self.replacement
                                                       )
             self.eec.fit(X_train,y_train)
             self.logger.log(self.file_object,
                                   'Balanced Ada Boost params: '+str(self.eec_random.best_params_)+'\t' + str(self.eec_random.best_score_)+'. Exited the get_best_params_for_AdaBoost method of the Model_Finder class')
             print('aba boost done and exited')
             return self.eec
         except Exception as e:
              self.logger.log(self.file_object,
                                   'Exception occured in get_best_params_for_balanced_adaBoost method of the Model_Finder class. Exception message:  ' + str(e))
              self.logger.log(self.file_object,
                                   'Balance Ada Boost tuning  failed. Exited the get_best_params_for_balanced_AdaBoost method of the Model_Finder class')
              raise Exception()
              
              
              
     def get_best_model(self,X_train, X_test, y_train, y_test):
         
                         
         self.logger.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        
         print('in get best model')
         try:
        
             self.brf= self.get_best_params_for_balanced_random_forest(X_train,y_train)
             self.y_pred_brf = self.brf.predict(X_test)
             self.brf_f2 = self.f2_make(y_test,self.y_pred_brf)



   
             self.eec= self.get_best_params_for_balanced_adaBoost(X_train,y_train)
             self.y_pred_eec = self.eec.predict(X_test)
             self.eec_f2 = self.f2_make(y_test,self.y_pred_eec)


            #comparing the two models
             if(self.brf_f2 >  self.eec_f2):
                 print('best model exited')
                 joblib.dump(self.brf, self.saved_best_model_path)
                 return 'BalancedRandomForestClassifier',self.brf
             else:
                 print('best model exited')
                 joblib.dump(self.eec, self.saved_best_model_path)
                 return 'EasyEnsembleClassifier',self.eec

         except Exception as e:
             self.logger.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
             self.logger.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
             raise Exception()              
             
a = Model_Finder()
X_train, X_test, y_train, y_test = a.X_train, a.X_test, a.y_train, a.y_test

#model_name, model = a.get_best_model(X_train, X_test, y_train, y_test)

model_loaded = joblib.load(a.saved_best_model_path)
y_pred = model_loaded.predict(X_test)
se = pd.Series(y_pred, name='y')
se1 = pd.Series(y_test, name='y')
pd.value_counts(se).plot(kind='bar')
pd.value_counts(se1).plot(kind='bar')
           
