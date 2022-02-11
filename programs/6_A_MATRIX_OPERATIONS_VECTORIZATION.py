import numpy as np

a=np.array([1,2,3])
print("Type : %s" %type(a))
print("Shape : %s" %a.shape)
print(a[0],a[1],a[2])
a[0]=5
print(a)

b=np.array([[1,2,3],[4,5,6]])
print("\nShape of b:",b.shape)
print(b[0,0],b[0,1],b[1,0])

a=np.zeros((2,2))
print("All zeros matrix : \n %s" %a)

b=np.ones((1,2))
print("\nAll ones matrix; \n %s" %b)

d=np.eye(2)
print("\n Identity matrix : \n %s" %d)

e = np.random.random((2,2))
print("\n random matrix: \n%s" %e)
print("vectorized sum example \n")

x = np.array([[1,2],[3,4]])
print("x:\n %s" %x)
print("sum: %s" %np.sum(x))
print("sum axis = 0 %s \n" %np.sum(x, axis=0))
print("sum axis = 1 %s \n" %np.sum(x, axis=1))


a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 

dp=np.dot(a,b)

print("Dot product; %s \n" %dp)


op=np.outer(a,b)
print("\nOuter product : %s\n" %op)



ep=np.multiply(a,b)
print("\n Element Wise product : %s \n" %ep)