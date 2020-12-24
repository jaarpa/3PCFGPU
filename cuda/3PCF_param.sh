#!/bin/bash

#Script para variar parametros del programa 3PCFisotropica

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

for ((dmax=20;dmax<=60;dmax+=10))
do 
	for ((partitions=20;partitions<=60;partitions+=5))
	do
			3PCFisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
	done	
done


# ------------------- Para caja de 1GPc --------------------------------------------

for ((dmax=20;dmax<=60;dmax+=10))
do 
	for ((partitions=20;partitions<=60;partitions+=5))
	do
			3PCFisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 50653 20 $dmax $partitions
	done	
done


# ------------------- Para caja de 2GpC --------------------------------------------

for ((dmax=20;dmax<=60;dmax+=10))
do 
	for ((partitions=20;partitions<=60;partitions+=5))
	do
			3PCFisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 50653 20 $dmax $partitions
	done	
done

