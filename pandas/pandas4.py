# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:30:21 2021

@author: Administrator
"""

import pandas as pd
s=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(s.iloc[1:4])
print(s.loc['a':'e'])
#print(s.loc[1:4])
#print(s.iloc['a':'e'])
