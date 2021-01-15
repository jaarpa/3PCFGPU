# 3PCFGPU

*Code capable to compute the isotropic or anisotropic, two or three Points Correlation Functions (PCF), with or without Boundary Periodic Conditions (BPC) from galaxy catalogs, using parallel programming on GPU or CPU.*

The code in this repository is capable to compute and vizualize the correlation functions of given galaxy catalogs. This repository contains two versions capable to produce the same results; in the cpp directory you can find the CPU programs to compute the PCF using openMP parallel programming, in the cuda folder the GPU programs are provided. 

## Prerequisites

#### CPU
The CPU code has been tested using the GCC compilers and everything you could need is shipped with the GCC compilers.

#### GPU
The GPU programs are developed using cuda and therefore a cuda capable GPU is required. If your computer has a capable GPU you must already have the nvidia `nvcc` compiler which you can do following [this instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Since histograms involved in the correlation function computations usually reach large countings this code uses some double precision functions, therefore yout GPU must have at least a 7.5 compute capability. You can check your GPU compute capability [here](https://developer.nvidia.com/cuda-gpus)

## Installation

Clone this repository using git or download the .zip folder from this webpage.
You can clone it by running `git clone https://github.com/jaarpa/DESI.git` in your terminal.

#### CPU

The CPU programs are compiled separately; in the cpp folder you will find four different folders one for each correlation function type, inside each of those four folders there will be a version with BPC, without BPC and in the isotropic versions an code to compute with BPC analytically. Each folder contains a main.cpp code which can be compiled with `c++ main.cpp -o main.out`

#### GPU

A master script has been provided for the easiest usage of the code. Place your terminal in the cuda folder of this repository and run `nvcc -arch=sm_75 main.cu -o PCF.out` to compile the master script, it may take a few seconds.

## Usage

The data files should be placed in the data/ folder or in a custom folder *inside* the data/ folder, this repository is shipped with some data files for example purposes. The data file must be formated as follows:

* Must have 4 columns, the first three are the positions of the point and the last column is the weight of the measurement.
* Every new point should be placed in a new row.

At first you have to place the data files you will use in the data/ folder. You can check the provided files in the data folder as an example.

Similarly every histogram result of the computations will be saved in the result folder, this repository is shipped with some result histograms for example purposes.

#### CPU

#### GPU



## Example

#### CPU

#### GPU

## Acknowledgments

The development of this code has profited from the very valuable input of a number of people. They are acknowledged here:

* [Gustavo Niz](https://github.com/gnizq64)
* [Fidel Sosa](https://github.com/fidelsosan)
* [Oscar De la Cruz Echeveste](https://github.com/Oscar2401)
* [Alejandro Gómez](https://github.com/AlejandroGoper)
* [Juan Antonio Aramburo Pasapera](https://github.com/jaarpa)

With funds providede by the Universidad de Guanajuato
