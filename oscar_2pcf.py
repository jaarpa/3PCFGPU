import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import time

start_0 = time.perf_counter()
d_max = 180
bins_number = 30

data = np.loadtxt(fname='fake_DATA/DATOS/data.dat', delimiter=" ", usecols=(0,1,2))
rand0 = np.loadtxt(fname='fake_DATA/DATOS/rand0.dat', delimiter=" ", usecols=(0,1,2))

#FUNCIÓN DE PARA HACER HISTOGRAMAS 
def Histos(p,p_r,bn,point_max):
    """ 
    Función para construir los histogramas 
    
    p = datos
    p_r = random
    bn = tamaño de bins
    point_max = punto máximo en el histograma
    
    """
    
    #Inicializamos los arreglos de los histogramas
    NDD = np.zeros(bn)
    NRR = np.zeros(bn)
    NDR = np.zeros(bn)
    
    n = 0
    
    start = time.perf_counter()
    for (ii, jj) in zip(p, p_r):
        n = n+1
        
        # Histogramas para DD
        s = ii-p[n:] # Diferencia entre el punto pivote y los demas puntos siguientes 
        dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bn, range=(0, point_max))
        NDD = NDD + 2*dis
        
        # Histogramas para RR
        s = jj-p_r[n:]
        dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bn, range=(0, point_max))
        NRR = NRR + 2*dis
        
    
    end = time.perf_counter()
    print(f'{end-start} for the RR and DD histogram')
    
    start = time.perf_counter()
    for ii in p:
        # Histogramas para DR
        s = ii-p_r
        dis, r = np.histogram(np.sqrt(s[:,0]**2+s[:,1]**2+s[:,2]**2), bins=bn, range=(0, point_max))
        NDR = NDR + dis
    
    end = time.perf_counter()
    print(f'{end-start} for the DR histogram')
    
    return r, NDD, NRR, NDR
    

#Función de correlación Landy-Szalay
def estim_LS(NDD, NRR, NDR):
    return (NDD - 2*NDR + NRR)/NRR

#def estim_LS(NDD, NRR, NDR, n, m, nm):
#    return 1 + (NDD*m)/(n*NRR) - 4*(NDR*m)/(nm*NRR)


#Función de correlación de Hamilton
def estim_HAM(NDD, NRR, NDR):
    return (NDD*NRR/NDR**2) - 1

#def estim_HAM(NDD, NRR, NDR, n, m, nm):
#    return (NDD*NRR*(nm**2))/((4*n*m)*NDR**2) - 1
    
start = time.perf_counter()

bins = 30
r, NDD, NRR, NDR = Histos(data,rand0,bins,180)

finish = time.perf_counter()

print(f'Finializó en {round(finish-start,2)} segundos')

eps_LS = estim_LS(NDD, NRR, NDR)
eps_HAM = estim_HAM(NDD, NRR, NDR)
