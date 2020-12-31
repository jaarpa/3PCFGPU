#!/bin/bash

#Script para variar parametros del programa 3PCFisotropica
#PID 16309

#Dmax --> [20..60..10] (de 20 a 60 en incrementos de 10)
#N_bins --> 20


touch time_results.dat #Abrimos archivo
nvcc -arch=sm_75 3PCFisotropic/normal/optimization.cu -o 3PCFisotropic/normal/opt.out

# ------------------- Encabezado del archivo ----------------------------------------

echo "3PCF isotropica. Sin BPC. Histograma DDD \n" >> time_results.dat
echo "caja 512MPc \n" >> time_results.dat
echo "n_points, size_box, d_max, partitions, time [s] \n" >> time_results.dat

# ------------------- Cuerpo del archivo --------------------------------------------

# ------------------- Para caja de 512MPc -------------------------------------------
partitions=25
for ((dmax=80;dmax<=120;dmax+=20))
do
    3PCFisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax -1 $partitions
done