import numpy as np
import matplotlib.pyplot as plt
 
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P
  
def dir_vec(A,B):
  return B-A

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
  
# enter vectors A,B & C
A=np.array([1,-1])
B=np.array([-4,6])
C=np.array([-3,-5])
# direction vector along line joining A & B
AB = dir_vec(A,B)
# direction vector along line joining A & C
AC = dir_vec(A,C)
# midpoint of A & B is F
F = (A+B)/2
# midpoint of A & C is E
E = (A+C)/2
# O is the point of intersection of perpendicular bisectors of AB and AC
O = line_intersect(AB,F,AC,E)
print(O)

#Generating all lines 
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)

#plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1.2 + 0.05), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1.2 + 0.05), B[1] * (1 - 0.1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1.2 + 0.05), C[1] * (1 - 0.1) , 'C')
plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1.2 + 0.05), O[1] * (1 - 0.1) , 'O')
plt.plot(E[0], E[1], 'o')
plt.text(E[0] * (1.2 + 0.05), E[1] * (1 - 0.1) , 'E')
plt.plot(F[0], F[1], 'o')
plt.text(F[0] * (1.2 + 0.05), F[1] * (1 - 0.1) , 'F')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
