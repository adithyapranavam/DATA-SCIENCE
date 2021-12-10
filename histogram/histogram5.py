import numpy as np
import matplotlib.pyplot as plt
ds=[1,11,21,31,41,51]
plt.hist(ds,bins=(0,10,20,30,40,50,60),weights=(10,1,0,33,6,8),facecolor='y',edgecolor='red')
plt.title("Histogram for students data")
plt.xlabel('value')
plt.ylabel('frequency')
plt.savefig("student.png")
plt.show()
    