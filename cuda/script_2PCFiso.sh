#!/usr/bin/bash

#=================================================================================================
# Script para correr de una sola vez todas las realizaciones de las cajas de 512 MPC
# Todas estas cajas contienen 50653 puntos y se correran a dmax = 140 MPc para 30 bins
#================================================================================================

for i in $(seq 0 1 14)
do
	./PCF.out 2iso -f data_512MPc_$i.dat -r data_512MPc_$i.dat -n 50653 -b 30 -d 140
done

echo "===================================================="
echo " 					       OK 						  "
echo "===================================================="
