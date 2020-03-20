# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:06:02 2020

@author: OF65
"""

from readWriteFile import readWriteOps
from application_logging.logger import App_Logger
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder

class dataTransform:

     """
               This class shall be used for transforming the Training and new predicion Data before loading it in Database!!.

               """

     def __init__(self):  
         self.file_object = open("../logs/datatransform/log.txt", 'a+')
         self.logger = App_Logger()
          
     def trainingData(self):
         self.logger.log(self.file_object,'Entered the trainingData method of the dataTransform class')
         try:
             data_getter=readWriteOps.Data_Getter()
             data = data_getter.get_data()
             df_filter = data.iloc[:, 3:]
             oe = OrdinalEncoder(dtype=np.int32)
             df_1 = oe.fit_transform(df_filter[['Geography', 'Gender']])
             df_2 = pd.DataFrame(data=df_1, columns=['Geography', 'Gender'])
             df_1= df_filter.drop(['Geography', 'Gender'], axis=1)
             df = pd.concat([df_2, df_1], axis=1)
             output = open('encoder.pkl', 'wb')
             pickle.dump(oe, output)
             output.close()
             self.logger.log(self.file_object,'Data transfomr Successful.Exited trainingData method of the dataTransform class')
             return df
         except Exception as e:
             self.logger.log(self.file_object,
                                   'Exception occured in trainingData method of the dataTransform class. Exception message: '+str(e))
             self.logger.log(self.file_object,
                                   'ataTransform Unsuccessful.Exited the trainingData method of the dataTransform class')

     
#ob1 = dataTransform()
#df = ob1.trainingData()
#df.head()
            
