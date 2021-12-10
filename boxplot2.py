import matplotlib.pyplot as plt
ds=[5,15,25,35,15,55]
plt.hist(ds,bins=(0,10,20,30,40,50,60),weights=(20,10,45,33,6,8),edgecolor='red')
plt.show()
    