#!/usr/bin/env bash

#Variar dmax [20,160]
#Variar caja -> npts [data_512MPc, 1GPc, 2GPC]
#bn=20
#SH PID: 1631

touch time_results.dat
nvcc -arch=sm_75 2PCFisotropic/normal/optimization.cu -o 2PCFisotropic/normal/opt.out

echo "2PCF isotropica. No BPC. Solo DDD \n" >> time_results.dat
echo "data_512MPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 10 3 26)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done
    for partitions in $(seq 26 44)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done

    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done
done
echo "\n \n" >> time_results.dat

echo "data_1GPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 10 3 26)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
    for partitions in $(seq 26 44)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
done
echo "\n \n" >> time_results.dat

echo "data_2GPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 15 3 24)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
    for partitions in $(seq 24 50)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
done
echo "\n \n" >> time_results.dat
echo "\n \n" >> time_results.dat

nvcc -arch=sm_75 2PCFanisotropic/normal/optimization.cu -o 2PCFanisotropic/normal/opt.out

echo "2PCF anisotropic. No BPC. Solo DDD. \n" >> time_results.dat
echo "data_512MPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 10 3 26)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done
    for partitions in $(seq 26 44)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done
    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_512MPc.dat data_512MPc.dat 50653 20 $dmax $partitions
        done
    done
done
echo "\n \n" >> time_results.dat

echo "data_1GPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 10 3 26)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
    for partitions in $(seq 26 44)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_1GPc.dat data_1GPc.dat 405224 20 $dmax $partitions
        done
    done
done
echo "\n \n" >> time_results.dat

echo "data_2GPc.dat \n" >> time_results.dat
echo "n_points, size_box, dmax, partitions, time [ms] \n" >> time_results.dat
for dmax in $(seq 20 20 160)
do 
    for partitions in $(seq 15 3 24)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
    for partitions in $(seq 24 50)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
    for partitions in $(seq 44 3 100)
    do 
        for i in $(seq 1 10)
        do
        2PCFanisotropic/normal/opt.out data_2GPc.dat data_2GPc.dat 3241792 20 $dmax $partitions
        done
    done
done
