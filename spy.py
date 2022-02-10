	<<plot a graph>>
import matplotlib.pyplot as plt
plt.plot([1,2,3],[5,7,4],linestyle="dashed")
plt.show()
	<<plot two graph>>
import matplotlib.pyplot as plt
x= [1,2,3]
y=[5,7,4]
plt.plot(x,y,label='First Line',linestyle="dashed")
x2=[1,2,3]
y2=[10,11,14]
plt.plot(x2,y2,label='Second Line')
plt.xlabel('plot number')
plt.ylabel('important variables')
plt.title('new graph')
plt.legend()
plt.show()
	<<plot twograph in same window>>
import matplotlib.pyplot as plt
import numpy as np
t=np.arange(0.0,20.0,1)
s=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
p=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
plt.subplot(2,1,1)
plt.plot(t,s)
plt.ylabel('value')
plt.title('first chart')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(t,p)
plt.xlabel('item(s)')
plt.ylabel('value')
plt.title('\n\n\nsecond')
plt.grid(True)
plt.show()
	<<list of numbers with index>>
import pandas as pd
s1=pd.Series([10,20,30,40,23],index=['first','second','third','fourth','fifth'])
s2=pd.Series(['a','b'])
print(s1)
s1=pd.Series(range(5))
print(s1)
	<<list>>
import pandas as pd
list=[10,20,30,40,50]
df1=pd.DataFrame(list)
print(df1)
	<<display list with default index>>
import pandas as pd1
d1=[['Anila',20],['ajay',22],['anu',18]]
df1=pd1.DataFrame(d1,columns=['Name','Age'])
print(df1)
	<<scatter plot>>
import matplotlib.pyplot as plt
import numpy as np
x=np.random.randn(1000)
y=np.random.randn(1000)
plt.scatter(x,y)
plt.show()
	<<bar chart>>
import numpy as np
import matplotlib.pyplot as plt
objects=('py','c++','java','perl','scala','lisp')
y_axis=np.arange(len(objects))
x_axis=objects
performance=[10,8,6,4,2,1]
plt.bar(x_axis,y_axis,width=.5,color='green')
plt.show()
	<<print all data set>>
import seaborn as sns
print(sns.get_dataset_names())
	<<swarm plot>>
import matplotlib.pyplot as plt
import seaborn as sns
ir=(sns.load_dataset('iris'))
sns.swarmplot(x="species",y="petal_length",data=ir)
plt.show()
	<<violin plot>>
import seaborn as ss
ss.set(style="whitegrid")
tips=ss.load_dataset("tips")
ss.violinplot(x="day",y="total_bill", data=tips)
	<<distribution plot>>
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
data = np.random.randn(200)
res = sn.distplot(data)
plt.show()
	<<barplot>>
import seaborn as sns
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.barplot(x="day", y="total_bill", data=tips)
	<<histogram>>
import matplotlib.pyplot as plt
import numpy as np
y=np.random.randn(1000)
plt.hist(y)
plt.show()
	<<histogram with edge colour>>
import matplotlib.pyplot as plt
import numpy as np
y=np.random.randn(1000)
plt.hist(y,25,edgecolor="orange")
plt.show()
	<<histogram with weight>>
import matplotlib.pyplot as plt
data_students=[5,15,25,35,45,55]
plt.hist(data_students,bins=[0,10,20,30,40,50,60],weights=[20,10,45,33,6,8],edgecolor="red")
plt.show()
	<<quartile with colour>>
import matplotlib.pyplot as plt
v1=[72,76,24,40,57,62,75,78,31,32]
v2=[62,5,91,25,36,32,96,95,30,90]
v3=[23,89,12,78,72,89,25,69,68,86]
v4=[99,73,70,16,81,61,88,98,10,87]
box_plot_data=[v1,v2,v3,v4]
box=plt.boxplot(box_plot_data,vert=1,patch_artist=True,labels=['c1','c2','c3','c4'],)
colors=['cyan','lightblue','lightgreen','tan']
for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color)
plt.show()
	<<quartile>>
import matplotlib.pyplot as plt
l1=[43,76,34,63,56,82,87,55,64,87,95,23,14,65,67,25,23,85]
l2=[34,45,34,23,43,76,26,18,24,74,23,56,23,23,34,56,32,23]
data=[l1,l2]
plt.boxplot(data)
plt.show()
	<<histogram of petal length>>
import matplotlib.pyplot as plt
import seaborn as sn
df = sn.load_dataset('iris')
sn.histplot(data=df,x='petal_length')
plt.show()
	<<scatter circle markers using bokeh>>
from bokeh.plotting import figure as fig  
from bokeh.plotting import output_notebook as ON  
from bokeh.plotting import show   
ON()   
plot1 = fig(plot_width = 500, plot_height = 500, title = "Scatter Markers")   
plot1.circle([1, 2, 3, 4, 5, 6], [2, 1, 6, 8, 0],   
size = 12, color = "green", alpha = 1)  
show(plot1)
	<<line using bokeh>>
from bokeh.plotting import figure  
from bokeh.plotting import output_notebook   
from bokeh.plotting import show  
output_notebook()  
plot1 = figure(plot_width = 500, plot_height = 500, title = "Scatter Markers")  
plot1.line([1, 2, 3, 4, 5, 6], [2, 1, 6, 8, 0],   
line_width = 4, color = "red")  
show(plot1) 
	<<square plot using bokeh>>
from bokeh.plotting import figure, output_notebook, show
output_notebook()
p = figure(width=400, height=400)
p.square([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="olive", alpha=0.5)
show(p)
	<<pie using bokeh>>
from bokeh.plotting import figure  
from bokeh.plotting import output_file   
from bokeh.plotting import show    
output_file("JTP.html")      
graph1 = figure(title = "Pie Chart using Bokeh")   
x = 0  
y = 0   
radius = 1   
start_angle = [0, 1.2, 2.1, 3.3, 5.1]
end_angle = [1.2, 2.1, 3.3, 5.1, 0]       
color1 = ["brown", "grey", "green", "orange", "red"]   
graph1.wedge(x, y, radius, start_angle, end_angle, color = color1)   
show(graph1) 
	<<change the position of toolbar in bokeh>>
from bokeh.plotting import figure, output_file, show
output_file("toolbar.html")
p = figure(width=400, height=400,title=None, toolbar_location="left",toolbar_sticky=False)
p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=10)
show(p) 
	<<segment wise display>>
from bokeh.plotting import figure, output_file, show
output_file("segment.html")
p = figure(width=400, height=400)
p.segment(x0=[1, 2, 3], y0=[1, 2, 3], x1=[1.2, 2.4, 3.1],y1=[1.2, 2.5, 3.7], color="#F4A582", line_width=3)
show(p)
