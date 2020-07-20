import matplotlib.pyplot as plt
import matplotlib.legend 
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.legend 
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numpy.polynomial import Polynomial, Legendre
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
cmap=plt.get_cmap('RdBu')
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
norm = MidpointNormalize(midpoint=0)

d_max=180
nb=30


#Creando histograma y su archivo de Posiciones
x=[]
DD=[]
DR=[]
RR=[]
for i in range(nb):
    x.append((i+1/2)*d_max/nb)
    DD.append(0)
    DR.append(0)
    RR.append(0)
x=np.array(x)
print(x)

#Importando datos
data=np.loadtxt('fake_DATA/DATOS/data.dat')
rand=np.loadtxt('fake_DATA/DATOS/rand0.dat')
N=len(data)
len(data)==len(rand)

#Definicion de una funcion para calcular la distancia entre dos puntos
def dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

start=timer()
print("Started the loop")
for i in range(N-1):
    #if int(i/100)==i/100:
        #print(i)
    for j in range(i,N):
        d_d=dist(data[i],data[j])
        d_r=dist(rand[i],rand[j])
        if d_d<d_max:
            DD[int(d_d*nb/d_max)]+=2
        if d_r<d_max:
            RR[int(d_r*nb/d_max)]+=2         
end=timer()
print(end-start)


start=timer()
for i in range(N):
    #if int(i/100)==i/100:
        #print(i)
    for j in range(N):
        d=dist(data[i],rand[j])
        if d<d_max:
            DR[int(d*nb/d_max)]+=1
end=timer()
print(end-start)