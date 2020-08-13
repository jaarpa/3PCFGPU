#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;

void main(){
    double end, start = omp_get_wtime();
    
    int i, nthreads=6;
    double pi, sum = 0.0;

    step = 1.0/(double) num_steps;
    omp_set_num_threads(nthreads);

    #pragma omp parallel 
    {
        int j, id;
        double x, sub_sum=0.0;
        id = omp_get_thread_num();
        if (id==0) {
            nthreads = omp_get_num_threads();
            printf("%d\n", nthreads);
            }
        for (j=id; j<num_steps; j=j+nthreads){
            x=(j+0.5)*step;
            sub_sum += 4.0/(1.0+x*x);
        }
        # pragma omp critical
        sum+=sub_sum;
    }
    pi = sum*step;
    printf("%f\n", pi);
    end = omp_get_wtime();
    printf("%f\n", end-start);

}