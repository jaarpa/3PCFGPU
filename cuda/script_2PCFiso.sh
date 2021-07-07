#!/usr/bin/bash

#=================================================================================================
# Script para correr de una sola vez todas las realizaciones de las cajas de 2 GPc
# Todas estas cajas contienen 3241792 puntos y se correran a dmax = 140 MPc para 30 bins
#================================================================================================

for i in $(seq 0 1 4)
do
	./PCF.out 2iso -f data_2GPc_$i.dat -r rand0_2GPc_$i.dat -n 3241792 -b 30 -d 140
done

echo "===================================================="
echo "                         OK                         "
echo "===================================================="
