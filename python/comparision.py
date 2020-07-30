import numpy as np
from iso_3pcf import pcf2_iso_histo, simetrize_3dhistogram
from oscar_iso_3pcf import Histos

data_location='../fake_DATA/DATOS/data_500.dat'
rand_location='../fake_DATA/DATOS/rand0_500.dat'

data = np.loadtxt(fname=data_location, delimiter=" ", usecols=(0,1,2))
rand0 = np.loadtxt(fname=rand_location, delimiter=" ", usecols=(0,1,2))

NDD_2d_2, NRR_2d_2, NDR_2d_2, xx_2, yy_2 = Histos(data,rand0,30,180)
RRR, DDD, DDR, RRD, edges = pcf2_iso_histo(data_location='../fake_DATA/DATOS/data_500.dat',rand_location='../fake_DATA/DATOS/rand0_500.dat', d_max=180.0, bins_number=30)