#include <stdio.h>
#include <omp.h>

static long num_steps = 100000;
double step;
int main(){
    double end, start = omp_get_wtime();
    int i; double pi, sum = 0.0;
    step = 1.0/(double) num_steps;
    #pragma omp parallel
    {
        double x;

        #pragma omp for reduction(+:sum)
            for (i=0; i<num_steps; i++){
                x = (i+0.5)*step;
                sum += 4.0/(1.0+x*x);
            }
    }
    pi = step*sum;
    printf("%f\n", pi);
    end = omp_get_wtime();
    printf("%f\n", end-start);

}
