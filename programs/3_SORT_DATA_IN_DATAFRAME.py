import pandas as pd
student_marks=pd.Series({'Vijaya':50,'Rahul':82,'Roopesh':97,'Athira':90,'Radhika':60})
student_age=pd.Series({'Vijaya':34,'Rahul':45,'Roopesh':30,'Athira':23,'Radhika':20})
student_df=pd.DataFrame({'Marks':student_marks,'Age':student_age})
print(student_df)
print(student_df.sort_values(by=['Marks']))
print(student_df.sort_values(by=['Marks'],ascending=False))