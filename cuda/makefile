objects = main.o help.o create_grid.o pcf2ani.o

all: $(objects)
	nvcc -arch=sm_60 $(objects) -o PCF.out

%.o: %.cu
	nvcc -arch=sm_60 -x cu -I. -dc $< -o $@

clean:
	rm -f *.o *.out