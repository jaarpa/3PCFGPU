import os
import numpy as np

from ani_cp_2pcf import pcf2_ani_cp_histo
from ani_z_2pcf import pcf2_ani_z_histo
from iso_2pcf import pcf2_iso_histo

files_dir = os.path.dirname(os.path.realpath(__file__))+'/fake_DATA/DATOS/'
files = os.listdir(files_dir)

#Assuming all the files state whether it is a file of random or data points.
random_files = [f for f in files if f.startswith('rand')] 
data_files = [f for f in files if f.startswith('data')] 
#Assuming all the files have the number of points at the en of the name
random_files.sort()
data_files.sort()

files = zip(data_files,random_files)

for data, rand in files:
    print(f'started with the file {data}')
    #isotropic
    print('started isotropic calculations')
    DD, RR, DR, bins = pcf2_iso_histo(data_location=files_dir+data,rand_location=files_dir+rand,d_max=180.0, bins_number=30)
    
    report_name = data[:-4] + "_DD_isotropic.dat"
    np.savetxt(report_name, DD)

    report_name = data[:-4] + "_RR_isotropic.dat"
    np.savetxt(report_name, RR)
    
    report_name = data[:-4] + "_DR_isotropic.dat"
    np.savetxt(report_name, DR)
    print('finished isotropic calculations')

    #anisotropic z axis
    print('started anisotropic calculations with z axis as reference')
    DD, RR, DR, x_edges, y_edges = pcf2_ani_z_histo(data_location=files_dir+data,rand_location=files_dir+rand,d_max=180.0, bins_number=30)
    
    report_name = data[:-4] + "_DD_anisotropic_z_axis.dat"
    np.savetxt(report_name, DD)

    report_name = data[:-4] + "_RR_anisotropic_z_axis.dat"
    np.savetxt(report_name, RR)
    
    report_name = data[:-4] + "_DR_anisotropic_z_axis.dat"
    np.savetxt(report_name, DR)
    print('finished anisotropic calculations with z axis as reference')

    #anisotropic observation point
    print('started anisotropic calculations with [125,125,1000000] observarion point')
    DD, RR, DR, x_edges, y_edges= pcf2_ani_cp_histo(data_location=files_dir+data,rand_location=files_dir+rand, observation_point=np.array([125,125,1000000]),d_max=180.0, bins_number=30)
    
    report_name = data[:-4] + "_DD_anisotropic_obs_point.dat"
    np.savetxt(report_name, DD)

    report_name = data[:-4] + "_RR_anisotropic_obs_point.dat"
    np.savetxt(report_name, RR)
    
    report_name = data[:-4] + "_DR_anisotropic_obs_point.dat"
    np.savetxt(report_name, DR)
    print('finished anisotropic calculations with [125,125,1000000] observarion point')