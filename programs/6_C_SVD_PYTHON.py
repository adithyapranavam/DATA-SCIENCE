from numpy import array
from scipy.linalg import svd
A=array([[1,2],[3,4],[5,6]])
print("A : \n %s" %A)
U,s,VT=svd(A)
print("\nU: \n%s" %U)
print("\ns: \n %s" %s)
print("\nV^T : \n%s" %VT)