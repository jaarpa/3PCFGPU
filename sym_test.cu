//nvcc sym_test.cu -o t.out && ./t.out data_5K.dat

//nvcc sym_test.cu -o t.out && ./t.out data_1GPc.dat
//Spent time = 0.0334 seg (ONly host. Nodes allocation and calculation. dataD in globalmemory )

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

/** CUDA check macro */
#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}

//Point with weight value. Structure

struct Point3D{
	float x;
	float y; 
	float z;
};

struct PointW3D{
    float x;
    float y; 
    float z;
    float w;
};

void save_histogram(string name, int bns, float *histo){
    /* This function saves a one dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    ofstream file;
    file.open(name.c_str(), ios::out | ios::binary);

    string mypathto_files = "data/";
    //This creates the full path to where I have my data files
    name.insert(0,mypathto_files);

    if (file.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            for (int k = 0; k < bns; k++){
                idx = i*bns*bns + j*bns + k;
                file << histo[idx] << ' ';
            }
            file << "\n";
        }
        file << "\n" << endl;
    }
    file.close();
}

void open_files(string name_file, int pts, PointW3D *datos, float &size_box){
    /* Opens the daya files. Receives the file location, number of points to read and the array of points where the data is stored */
    ifstream file;

    string mypathto_files = "data/";
    //This creates the full path to where I have my data files
    name_file.insert(0,mypathto_files);

    file.open(name_file.c_str(), ios::in | ios::binary); //Tells the program this is a binary file using ios::binary
    if (file.fail()){
        cout << "Failed to load the file in " << name_file << endl;
        exit(1);
    }

    double candidate_size_box=0;
    double max_component;
    for ( int c = 0; c < pts; c++) //Reads line by line and stores each c line in the c PointW3D element of the array
    {
        file >> datos[c].x >> datos[c].y >> datos[c].z >> datos[c].w;

        if (datos[c].x>datos[c].y){
            if (datos[c].x>datos[c].z){
                max_component = datos[c].x;
            } else {
                max_component = datos[c].z;
            }

        } else {
            if (datos[c].y>datos[c].z){
                max_component = datos[c].y;
            } else {
                max_component = datos[c].z;
            }
        }

        if (max_component>candidate_size_box){
            candidate_size_box = max_component;
        }
    }

    size_box=ceil(candidate_size_box+1);

    file.close();
}


__global__ void simmetrization(float *s_XXX,float *XXX , int bn){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v;
    printf("%d \n", XXX[i]);
    for (int j=i+1; j<bn; j++){
        for (int k=j+1; k<bn; k++){
            v = XXX[i*bn*bn + j*bn + k] + XXX[i*bn*bn + k*bn + j] + XXX[j*bn*bn + k*bn + i] + XXX[j*bn*bn + i*bn + k] + XXX[k*bn*bn + i*bn + j] + XXX[k*bn*bn + j*bn + i];
            s_XXX[i*bn*bn + j*bn + k] = v;
            s_XXX[i*bn*bn + k*bn + j] = v;
            s_XXX[j*bn*bn + k*bn + i] = v;
            s_XXX[j*bn*bn + i*bn + k] = v;
            s_XXX[k*bn*bn + i*bn + j] = v;
            s_XXX[k*bn*bn + j*bn + i] = v;
        }
    }
    
}

int main(int argc, char **argv){
    unsigned int np = 10000;
    int bn=20;
    float size_box = 0;
    clock_t start_timmer, stop_timmer;
    double time_spent;

    float *DDD, *d_DDD, *sd_DDD;
    DDD = new float[bn*bn*bn];
    cucheck(cudaMallocManaged(&d_DDD, bn*bn*bn*sizeof(float)));
    cucheck(cudaMallocManaged(&sd_DDD, bn*bn*bn*sizeof(float)));

    PointW3D *dataD;
    dataD = new PointW3D[np];
    open_files(argv[1], np, dataD, size_box);

    for (int i=0; i<bn*bn*bn; i++){
        d_DDD[i] = dataD[i].x+dataD[i].y-dataD[i].z;
    }

    for (int i=0; i<12; i++){
        cout << d_DDD[i] <<endl;
    }
    cout << "Entering to the device code "<< endl;
    //cucheck(cudaMemcpy(d_DDD, DDD, bn*bn*bn*sizeof(float), cudaMemcpyHostToDevice));

    start_timmer = clock();

    simmetrization<<<1,bn>>>(sd_DDD, d_DDD, bn);
    cucheck(cudaDeviceSynchronize());
    stop_timmer = clock();

    time_spent = (double)(stop_timmer - start_timmer) / CLOCKS_PER_SEC;
    printf("\nSpent time = %.4f seg.\n", time_spent );

    //cucheck(cudaMemcpy(DDD, sd_DDD, bn*bn*bn*sizeof(float), cudaMemcpyDeviceToHost));
    
    save_histogram("DDD.dat", bn, d_DDD);
    
    delete[] dataD;
    delete[] DDD;
    cucheck(cudaFree(d_DDD));
    
    cout << "Finished" << endl;
    return 0;

}
