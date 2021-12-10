# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:28:23 2021

@author: Administrator
"""

import pandas as pd
stmark=pd.Series({'akd':45,'qwe':50,'vk':32})
stage=pd.Series({'akd':21,'qwe':20,'vk':19})
print(stage)
stdf1=pd.DataFrame({'Marks':stmark,'Age':stage})
print(stdf1)
