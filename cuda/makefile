objects = main.o help.o create_grid.o

all: $(objects)
	nvcc $(objects) -o PCF.out

%.o: %.cu
	nvcc -x cu -I. -dc $< -o $@

clean:
	rm -f *.o *.out