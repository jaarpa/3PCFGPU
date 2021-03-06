#include <iostream>
#include <fstream>
#include <string.h>
#include <iomanip>

using namespace std;

struct Point3D{
	float x;
	float y; 
	float z;
};

//Point with weight value. Structure
struct PointW3D{
    float x;
    float y; 
    float z;
    float w;
};

struct Node{
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    PointW3D *elements;	// Points in the node
};

struct DNode{ //Defines the node in the device without using elements to avoid deep copy
    Point3D nodepos; //Position of the node
    int len;		// Number of points in the node
    int start; //prev element idx
    int end; //last element idx [non inclusive]
};

void open_files(string name_file, int pts, PointW3D *datos, float &size_box){
    /* Opens the daya files. Receives the file location, number of points to read and the array of points where the data is stored */
    ifstream file;

    string mypathto_files = "../data/";
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

    if (size_box<=0) size_box=ceil(candidate_size_box+1);

    file.close();
}

//=================================================================== 
void add(PointW3D *&array, int &lon, float _x, float _y, float _z, float _w){
    /*
    This function manages adding points to an specific Node. It receives the previous array, longitude and point to add
    and updates the previous array and length with the same array with the new point at the end and adds +1 to the length +1

    It manages the memory allocation and free of the previous and new elements.
    */
    lon++;
    PointW3D *array_aux = new PointW3D[lon];
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
        array_aux[i].w = array[i].w;
    }
    delete[] array;
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y;
    array[lon-1].z = _z;
    array[lon-1].w = _w;
}

void make_nodos(Node ***nod, PointW3D *dat, unsigned int partitions, float size_node, unsigned int np){
    /*
    This function classifies the data in the nodes

    Args
    nod: Node 3D array where the data will be classified
    dat: array of PointW3D data to be classified and stored in the nodes
    partitions: number nodes in each direction
    size_node: dimensions of a single node
    np: number of points in the dat array
    */

    int row, col, mom;

    // First allocate memory as an empty node:
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){
                nod[row][col][mom].nodepos.z = ((float)(mom)*(size_node));
                nod[row][col][mom].nodepos.y = ((float)(col)*(size_node));
                nod[row][col][mom].nodepos.x = ((float)(row)*(size_node));
                nod[row][col][mom].len = 0;
                nod[row][col][mom].elements = new PointW3D[0];
            }
        }
    }

    // Classificate the ith elment of the data into a node and add that point to the node with the add function:
    for (int i=0; i<np; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z, dat[i].w);
    }
}

//=================================================================== 
void save_histogram1D(string name, int bns, double *histo, int nhistos=0){
    /* This function saves a 1 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    string mypathto_files = "../results/";
    //This creates the full path to where I save the histograms files
    name.insert(0,mypathto_files);

    ofstream file2;
    file2.open(name.c_str(), ios::out | ios::binary);

    if (file2.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }
    
    for (int i = 0; i < bns; i++){
        file2 << setprecision(12) << histo[nhistos*bns + i] << endl;
    }

    file2.close();
}

//====================================================================
void save_histogram2D(string name, int bns, double *histo, int nhistos=0){
    /* This function saves a 2 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */
    
    string mypathto_files = "../results/";
    //This creates the full path to where I have my data files
    name.insert(0,mypathto_files);

    ofstream file;
    file.open(name.c_str(), ios::out | ios::binary);

    if (file.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            idx = nhistos*bns*bns + i*bns + j;
            file << setprecision(12) << histo[idx] << ' ';
        }

		file << "\n";
    }
    file.close();
}

//====================================================================
void save_histogram3D(string name, int bns, double *histo, int nhistos=0){
    /* This function saves a 3 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    string mypathto_files = "../results/";
    //This creates the full path to where I have my data files
    name.insert(0,mypathto_files);

    ofstream file;
    file.open(name.c_str(), ios::out | ios::binary);

    if (file.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            for (int k = 0; k < bns; k++){
                idx = nhistos*bns*bns*bns + i*bns*bns + j*bns + k;
                file << setprecision(12) << histo[idx] << ' ';
            }
            file << "\n";
        }
        file << "\n" << endl;
    }
    file.close();
}

//====================================================================
void save_histogram5D(string name, int bns, double *histo, int nhistos=0){
    /* This function saves a 5 dimensional histogram in a file.
    Receives the name of the file, number of bins in the histogram and the histogram array
    */

    string mypathto_files = "../results/";
    //This creates the full path to where I have my data files
    name.insert(0,mypathto_files);
    
    ofstream file;
    file.open(name.c_str(), ios::out | ios::binary);

    if (file.fail()){
        cout << "Failed to save the the histogram in " << name << endl;
        exit(1);
    }

    int idx;

    for (int i = 0; i < bns; i++){
        for (int j = 0; j < bns; j++){
            for (int k = 0; k < bns; k++){
                for (int l = 0; l < bns; l++){
                    for (int m = 0; m < bns; m++){
                        idx = nhistos*bns*bns*bns*bns*bns + i*bns*bns*bns*bns + j*bns*bns*bns + k*bns*bns + l*bns + m;
                        file << setprecision(12) << histo[idx] << ' ';
                    }
                    file << "\n";
                }
                file << "\n";
            }
            file << "\n";
        }
        file << "\n" << endl;
    }
    file.close();
}
