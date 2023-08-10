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

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
O = O.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
tri_coords = np.block([[A,B,C,O,E,F]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
