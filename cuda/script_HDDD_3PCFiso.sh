#!/usr/bin/bash

#=================================================================================================
# Script para analizar el tiempo de computo del histograma DDD de la 3PCF isotropica sin BPC
# para la caja 0 de 1 GPc 
#================================================================================================

for dmax in $(seq 30 10 180)
do
	./PCF.out 3iso -f data_1GPc_0.dat -r data_1GPc_0.dat -n 405224 -b 30 -d $dmax
done

echo "===================================================="
echo " 						OK                            "
echo "===================================================="
