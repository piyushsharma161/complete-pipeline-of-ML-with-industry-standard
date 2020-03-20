# -*- coding: utf-8 -*-

import pandas as pd
from application_logging.logger import App_Logger

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: Piyush
    Version: 1.0
    Revisions: None

    """
    def __init__(self):
        self.training_file='../training_file/Churn_Modelling.csv'
        self.file_object = open("../logs/filereadlogs/log.txt", 'a+')
        self.logger = App_Logger()

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: Piyush
        Version: 1.0
        Revisions: None

        """
        self.logger.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            self.data= pd.read_csv(self.training_file) # reading the data file
            self.logger.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.logger.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logger.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()
    

