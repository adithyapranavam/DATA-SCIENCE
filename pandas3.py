# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:11:54 2021

@author: Administrator
"""

import pandas as pd
s=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(s)
print("0th index value:",s[0])
print(s[:3])
print(s[-3:])
print(s['a'])
print(s['e'])
print(s[-1: :-1])
print(s['e': :-1])