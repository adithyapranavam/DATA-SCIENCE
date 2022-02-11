import matplotlib.pyplot as pll
objects=('java','python','javascript','c#','c++')
p=(22.2,8.8,8,77,6.7)
pll.bar(objects,p,align='center',alpha=0.4,color='blue')
pll.ylabel('popularity')
pll.xlabel('programming language')
pll.title('programming language popularity')
pll.show()
