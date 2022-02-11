import matplotlib.pyplot as pll
objects=('java','python','javascript','c#','c++')
p=(22.2,8.8,8,77,6.7)
pll.barh(objects,p,align='center',alpha=0.4,color=['red','blue','cyan','black','yellow'])
pll.xlabel('popularity')
pll.ylabel('programming language')
pll.title('programming language popularity')
pll.show()
