#include <stdio.h>
#include "help.cuh"

void show_help(){
    printf("Use: ./PCF <calc_type> -n sample_size -f datafile -b bins -d maximum_distance \n");
    printf("-r randomfile || -rd randoms_directory \n");
    printf("[-bpc [-a]][-p box_partitions] [-sb size_box] \n");

    printf(
        "Calculate the correlation function of 2 or 3 points either isotropic or anisotropic with either bondary periodic conditions or without. \n"
        "The files must contain 4 columns, the first 3 with the x,y,z coordinates and the 4 the weight of the measurment. \n"
        "Every file must contain at least the number of points specified in -p argument. \n"
        "Only one of the parameters, either -r or -rd must be provided. \n \n \n"
    );

    printf(
        "About the boundary periodic conditions:  \n"
        "If it is required to consider boundary periodic conditions in any of the calculation types add the -bpc option. \n"
        "If the bpc is followed by its -a (analytic) option neither the -r or the -rd options are required and will be ignored if found. \n"
        "If bpc are required but a size of the box is not provided with the -sb argument the program will compute the size of the box based \n"
        "on the maximum value of the position of all the points. This may lead to non consistent results \n \n \n"
    );

    printf(
        "<calc_type> options: \n \n"
        "3iso: Isotropic 3 points correlation function. \n"
        "3ani: Anisotropic 3 points correlation function.\n"
        "2iso: Isotropic 2 points correlation function. \n"
        "2ani: Anisotropic 2 points correlation function. \n"
    );

    printf(
        "Parameters: \n \n"
        "-n sample size:        sample size chosen randomply from the provided data file. \n\n"
        "-f datafile:           (required) datafile must be the name of a plain text file located in the data/ directory. The file shall contain \n"
        "                       3 columns for the x,y,z and a fourth column for the wight of each point. The file must contain at least \n"
        "                       the number of row specified by the -p parameter. \n \n"
        "-b bins:               (required) Bins must be an integer of the number of bins per dimension of the histogram. \n\n"
        "-d maximum_distance:   (required) Maximum_distance must be a real number with the maximum distance of interest between the points. \n\n"
        "-r randomfile:         (required) randomfile must be the name of a plain text file located in the data/ directory. The file shall contain \n"
        "                       3 columns for the x,y,z and a fourth column for the wight of each random point. The file must contain at \n"
        "                       least the number of row specified by the -p parameter. \n\n"
        "-rd randoms_directory: (required) randoms_directory shall be a folder with all the random files which will be used to compare the data files \n"
        "                       the computation of every histogram that requires a random file will be repeated for each file. \n\n"
        "-bpc:                  (optional) Add this option to request the boundary periodic conditions consideration. \n"
        "  [-a]:                (optional) Add this option to the -bpc option for the RRR and XXY histogras to be computed analytically. \n"
        "                       this option is only available for isotropic PCF computations.\n\n"
        "-p box_partitions:     (optional) Add this option to specify the number of partitions per grid dimension. \n\n"
        "-sb size_box:           (optional) size_box must be a real number which specifies the size of the box containing the the points. \n"
        "                       This option is recomended if -bpc is used to obtain consistent results. \n \n");
}