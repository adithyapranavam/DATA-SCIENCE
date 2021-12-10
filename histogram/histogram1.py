import matplotlib.pyplot as plt
d=[5,15,25,35,45,55]
plt.hist(d,bins=[0,10,20,30,40,50,60],weights=[20,10,45,33,6,8],histtype="step",edgecolor="red")
plt.show()
