#!/usr/bin/env bash

#Variar dmax [20,160]
#Variar caja -> npts [data_512MPc, 1GPc, 2GPC]
#bn=20

#Tamaño del nodo se varia dentro del .cu dependiendo del size_box y dmax

touch time_results.dat
nvcc optimization.cu -o opt.out
#c++ main.cpp -o serial.out
echo "data_2GPc.dat \n" >> time_results.dat
./par_s.out data_2GPc.dat dummy 3241792 20 $dmax
echo "\n \n" >> time_results.dat

echo "data_1GPc.dat \n" >> time_results.dat
./par_s.out data_1GPc.dat dummy 405224 20 $dmax
echo "\n \n" >> time_results.dat

echo "data_512MPc.dat \n" >> time_results.dat
./par_s.out data_512MPc.dat dummy 50653 20 $dmax

# Imprimir en time_results...
#File
#n_points, size_box, dmax, partitions, node_size, time [s]