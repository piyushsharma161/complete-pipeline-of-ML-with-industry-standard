# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:51:01 2020

@author: OF65
"""

from readWriteFile import readWriteOps



data_getter=readWriteOps.Data_Getter()

df = data_getter.get_data()
df.head()
