# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:59:22 2020

@author: OF65
"""

import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import App_Logger

class dBOperation:
    """
      This class shall be used for handling all the SQL operations.


      """
    def __init__(self):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()