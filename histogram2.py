import matplotlib.pyplot as plt
import seaborn as sb
iris=sb.load_dataset("iris")
sb.histplot(data=iris,x="petal_length")
plt.show()
