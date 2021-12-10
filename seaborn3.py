# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:16:30 2021

@author: Administrator
"""

import seaborn as sb
import matplotlib.pyplot as p
ir=sb.load_dataset("iris")
#print(ir)
sb.swarmplot(x="species",y="petal_length",data=ir)
p.show()