# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:46:52 2021

@author: Administrator
"""

import pandas as pd 
s1=pd.Series([10,20,30,40,50],index=['a','b','c','d','e'])
print(s1.head(2))
print(s1.head(5))
print(s1.tail(6))
print(s1.tail(3))