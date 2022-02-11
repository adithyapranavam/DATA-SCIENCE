import pandas as pd
student={'Unit test-1':[5,6,8,3,10],'Unit Test-2':[7,8,9,6,15]}
student1={'Unit test-1':[3,3,6,6,8],'Unit Test-2':[5,9,8,10,5]}
ds=pd.DataFrame(student)
ds1=pd.DataFrame(student1)
print(ds)
print(ds1)
print("Addition")
print(ds.add(ds1))
print("Subtration")
print(ds.sub(ds1))
print("Multiplication")
print(ds.mul(ds1))
print("Division")
print(ds.div(ds1))