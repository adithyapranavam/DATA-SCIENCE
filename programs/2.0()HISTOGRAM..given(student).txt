import matplotlib.pyplot as plt
data_students=(5,15,25,35,45,55)
plt.hist(data_students,bins=[0,10,20,30,40,50,60],weights=[20,10,45,33,6,8],edgecolor='red')
plt.xlabel('No of students')
plt.ylabel('Marks')
plt.show()