/* gcc -fopenmp foo.c */
/* export OMP_NUM_THREADS=4 */

#include <stdio.h>
#include <omp.h>

int main(){
    #pragma omp parallel 
    {
        int ID = omp_get_thread_num();
        
        printf("Numero de threads(%d)",omp_get_num_threads());
        printf("hello(%d)",ID);
        printf("world(%d)\n",ID);
    }
}
