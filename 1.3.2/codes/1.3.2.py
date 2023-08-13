import numpy as np
import matplotlib.pyplot as plt
  
#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 


def dir_vec(A,B):
  return B-A

#Foot of the Altitude
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P

  
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

#normal vector of AD_1 is n
n=dir_vec(B,C)

print(n,"x","=",n@A)

D_1 = alt_foot(A,B,C)

#Generating all lines 
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD_1 = line_gen(A,D_1)

#plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD_1$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D_1 = D_1.reshape(-1,1)

tri_coords = np.block([[A,B,C,D_1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','$D_1$']
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
