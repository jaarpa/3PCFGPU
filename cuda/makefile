all: a.out

a.out: main.o create_grid.o help.o
	gcc main.o create_grid.o help.o

main.o: main.c create_grid.h help.h
	gcc -c main.c

create_grid.o: create_grid.c create_grid.h
	gcc -c create_grid.c

help.o: help.c help.h
	gcc -c help.c

clean:
	rm main.o create_grid.o help.o