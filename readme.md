# 3PCFGPU

*Code capable to compute the isotropic or anisotropic, two or three Points Correlation Functions (PCF), with or without Boundary Periodic Conditions (BPC) from galaxy catalogs, using parallel programming on GPU or CPU.*

The code in this repository is capable to compute and vizualize the correlation functions of given galaxy catalogs. This repository contains two versions capable to produce the same results; in the cpp directory you can find the CPU programs to compute the PCF using openMP parallel programming, in the cuda folder the GPU programs are provided. 

## Prerequisites

#### CPU
The CPU code has been tested using the GCC compilers and everything you could need is shipped with the GCC compilers.

#### GPU
The GPU programs are developed using cuda and therefore a cuda capable GPU is required. If your computer has a capable GPU you must already have the nvidia `nvcc` compiler which you can do following [this instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Since histograms involved in the correlation function computations usually reach large countings this code uses some double precision functions, therefore yout GPU must have at least a 6.0 compute capability or higher. You can check your GPU compute capability [here](https://developer.nvidia.com/cuda-gpus)

## Installation

Clone this repository using git or download the .zip folder from this webpage.
You can clone it by running `git clone https://github.com/jaarpa/DESI.git` in your terminal.

#### CPU

The CPU programs are compiled separately; in the cpp folder you will find four different folders one for each correlation function type, inside each of those four folders there will be a version with BPC, without BPC and in the isotropic versions an code to compute with BPC analytically. Each folder contains a main.cpp code which can be compiled with `c++ main.cpp -o main.out`

#### GPU

A master script has been provided for the easiest usage of the code. Place your terminal in the cuda folder of this repository and run 
```
nvcc -arch=sm_60 main.cu -o PCF.out
``` 
to compile the master script, it may take a few seconds. If your GPU has a compute capability higher than 6.0 use its compute capability in the -arch option as -arch=sm_XY to get a better compiler optimization.

## Usage

The data files should be placed in the data/ folder or in a custom folder *inside* the data/ folder, this repository is shipped with some data files for example purposes. The data file must be formated as follows:

* Must have 4 columns, the first three are the coordinates of the point and the last column is the weight of the measurement.
* Every new point should be placed in a new row.
* No rows with any other formatting are allowed and could lead to unexpected behaviour.

At first you must to place the data (and random if required) files in the data/ folder. You can check the provided files in the data folder as an example.

Similarly every histogram result of the computations will be saved in the result folder, this repository is shipped with some result histograms for example purposes.

#### CPU

(In construction)

#### GPU
The master script must be placed in the cuda/ folder. You can run `./PCF --help` to get information about how to run this master script.

The the compiled PCF program receives command line arguments to decide what kind of correlation function is requested and where to get the data files from. The first argument it is always what kind of correlation function histograms you want to compute, the available options are:

* 3iso: Three points isotropic correlation function.
* 3ani: Three points anisotropic correlation function.
* 2iso: Two points isotropic correlation function.
* 3ani: Two points anisotropic correlation function.

The next arguments can be placed in any order and must be placed as -arg_keychar value. The next required arguments are:

* -n number_of_points: The number of points that will be used from the files to compute the histograms. **The files must contain at least number_of_points rows.**
* -f path_to_datafile: The path to the file containing the measurements to compute the histograms. **The path must be relative to the data/ directory.**
* -r path_to_randfile: The path to the file containing random points to compute the histograms. **The path must be relative to the data/ directory.**
* -rd path_to_randomfiles_directory: The path to a directory containing several random files. The histograms will be computed fore every random file in this directory vs the data file. Only one of the -r and -rd parameters must be provided.
* -b number_of_bins: The number of bins per histogram dimension. The number of bins may affect notoriously the time of computation.
* -d max_distance: The max distance of interest, if any pair of points is separated by a larger distance than max_distance that triangle/pair wont be counted.

The optional argumets for a finer tuning are:

* -bpc: This option activates the boundary periodic conditions computations.
  * -a: This activates the analytic computation of the DRR, DDR, RRR. This option is only compatible with the isotropic point correlation functions with BPC. If the -bpc -a parameters combination is provided the -r or -rd are not required.
* -p box_partitions: Explicitly set the number of partitions per grid dimension.
* -s size_box: Explicitly set the size of the box that contains the points. Every data and random file is assume to have the same size box. **The size_box must be larger than any point position component.**

## Examples

The following commands use the provided data files in this repository and assumes the compilation has been performed as in the Installation section.

#### CPU
(In construction)

#### GPU

To compute the three points isotropic correlation function histograms with 30 bins, a maximum interest distance of 80 with boundary periodic conditions analytically with 32768 points in a 250 side box run:
```
./PCF.out 3iso -bpc -a -f data.dat -n 32768 -b 30 -d 80 -s 250
```

To compute the three points isotropic correlation function with 30 bins a maximum interest distance of 80 with no boundary periodic conditions with 32768 points run:
```
./PCF.out 3iso -f data.dat -r rand0.dat -n 32768 -b 30 -d 80
```

To compute the three points anisotropic correlation function with 30 bins a maximum interest distance of 50 with boundary periodic conditions with 32768 points run:
```
./PCF.out 3ani -f data.dat -r rand0.dat -n 32768 -b 30 -d 80 -bpc
```

You will find the results in the results/ directory.

## Acknowledgments

The development of this code has profited from the very valuable input of a number of people. They are acknowledged here:

* [Aramburo Pasapera Juan Antonio](https://github.com/jaarpa)
* [De la Cruz Echeveste Oscar](https://github.com/Oscar2401)
* [Gómez Pérez Alejandro](https://github.com/AlejandroGoper)
* [Niz Quevedo Gustavo](https://github.com/gnizq64)
* [Sosa Nuñez Fidel](https://github.com/fidelsosan)

With funds providede by the Universidad de Guanajuato
