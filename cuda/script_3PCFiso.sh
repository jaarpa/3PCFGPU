#!/usr/bin/bash

#=================================================================================================
# Script para correr de una sola vez todas las realizaciones de las cajas de 512 MPc
# Todas estas cajas contienen 50653 puntos y se correran a dmax = 140 MPc para 30 bins
#================================================================================================

for i in $(seq 0 1 9)
do
	./PCF.out 3iso -f data_1GPc_$i.dat -r rand0_1GPc_$i.dat -n 405224 -b 30 -d 140
done

echo "===================================================="
echo "                         OK                         "
echo "===================================================="
