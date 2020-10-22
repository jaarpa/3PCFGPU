// nvcc distances.cu -o par.out && ./par.out data.dat rand0.dat 32768 30 180
#include<iostream>
#include<fstream>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

using namespace std;

//Structura que define un punto 3D
//Accesa a cada componente con var.x, var.y, var.z
struct Punto{
    double x,y,z;
};

struct Node{
    //Punto nodepos;	// Coordenadas del nodo (posición del nodo) // Se obtiene con las coordenadas del nodo.
    //int in_vicinage;    //Cantidad de nodos vecinos.
    //int *nodes_vicinage;     // Array con los master id de localizacion de los nodos vecinos.
    int len;		// Cantidad de elementos en el nodo.
    Punto *elements;
};

void read_file(string file_loc, Punto *data){
    //cout << file_loc << endl;
    string line; //No uso esta variable realmente, pero con eof() no se detenía el loop
    
    ifstream archivo(file_loc);
    
    if (archivo.fail() | !archivo ){
        cout << "Error al cargar el archivo " << endl;
        exit(1);
    }
    
    
    int n_line = 1;
    if (archivo.is_open() && archivo.good()){
        archivo >> data[0].x >> data[0].y >> data[0].z;
        while(getline(archivo, line)){
            archivo >> data[n_line].x >> data[n_line].y >> data[n_line].z;
            n_line++;
        }
    }
    //cout << "Succesfully readed " << file_loc << endl;
}

void save_histogram(string name, int bns, unsigned int ***histo){
    int i, j, k;
    unsigned int **reshape = new unsigned int*[bns];
    for (i=0; i<bns; i++){
        *(reshape+i) = new unsigned int[bns*bns];
        }
    for (i=0; i<bns; i++){
    for (j=0; j<bns; j++){
    for (k=0; k<bns; k++){
        reshape[i][bns*j+k] = histo[i][j][k];
    }
    }
    }
    ofstream file;
    file.open(name.c_str(),ios::out | ios::binary);
    if (file.fail()){
        cout << "Error al guardar el archivo " << endl;
        exit(1);
    }
    for (i=0; i<bns; i++){
        for (j=0; j<bns*bns; j++){
            file << reshape[i][j] << " "; 
        }
        file << endl;
    }
    file.close();
}

__device__
void count_3_N111(Punto *elements, unsigned int len, unsigned int ***XXX, float dmax2, float ds, bool t){
    /*
    Funcion para contar los triangulos en un mismo Nodo.

    row, col, mom => posición del Nodo. Esto define al Nodo.

    */

    unsigned int i,j,k;
    unsigned int a,b,c;
    
    float dx,dy,dz;
    float d12,d13,d23;
    float x1,y1,z1,x2,y2,z2,x3,y3,z3;

    for (i=0; i<len-2; ++i){
        x1 = elements[i].x;
        y1 = elements[i].y;
        z1 = elements[i].z;
        for (j=i+1; j<len-1; ++j){
            x2 = elements[j].x;
            y2 = elements[j].y;
            z2 = elements[j].z;
            dx = x2-x1;
            dy = y2-y1;
            dz = z2-z1;
            d12 = dx*dx+dy*dy+dz*dz;
            if (d12<=dmax2){
                d12 = sqrt(d12);
                for (k=j+1; k<len; ++k){ 
                    x3 = elements[k].x;
                    y3 = elements[k].y;
                    z3 = elements[k].z;
                    dx = x3-x1;
                    dy = y3-y1;
                    dz = z3-z1;
                    d13 = dx*dx+dy*dy+dz*dz;
                    if (d13<=dmax2){
                        d13 = sqrt(d13);
                        dx = x3-x2;
                        dy = y3-y2;
                        dz = z3-z2;
                        d23 = dx*dx+dy*dy+dz*dz;
                        if (d23<=dmax2){
                            d23 = sqrt(d23);
                            a = (unsigned int)(d12*ds);
                            b = (unsigned int)(d13*ds);
                            c = (unsigned int)(d23*ds);
                            if (t==1){
                                printf("Got to the end \n");
                            }
                            //atomicAdd(&XXX[1][2][3],1);
                        }
                    }
                }
            }
        }
    }
}

__device__
void count_3_N112(Punto *elements1, unsigned int len1, Punto *elements2, unsigned int len2, unsigned int ***XXX, float dmax2, float ds, float size_node){
    /*
    Funcion para contar los triangulos en dos 
    nodos con dos puntos en N1 y un punto en N2.

    row, col, mom => posición de N1.
    u, v, w => posición de N2.

    */
    int i,j,k;
    unsigned int a,b,c;
    float dx,dy,dz;
    float d12,d13,d23;
    float x1,y1,z1,x2,y2,z2,x3,y3,z3;

    for (i=0; i<len2; ++i){
        // 1er punto en N2
        x1 = elements2[i].x;
        y1 = elements2[i].y;
        z1 = elements2[i].z;
        for (j=0; j<len1; ++j){
            // 2do punto en N1
            x2 = elements1[j].x;
            y2 = elements1[j].y;
            z2 = elements1[j].z;
            dx = x2-x1;
            dy = y2-y1;
            dz = z2-z1;
            d12 = dx*dx+dy*dy+dz*dz;
            if (d12<=dmax2){
                d12=sqrt(d12);
                for (k=j+1; k<len1; ++k){
                    // 3er punto en N1
                    x3 = elements1[k].x;
                    y3 = elements1[k].y;
                    z3 = elements1[k].z;
                    dx = x3-x1;
                    dy = y3-y1;
                    dz = z3-z1;
                    d13 = dx*dx+dy*dy+dz*dz;
                    if (d13<=dmax2){
                        d13 = sqrt(d13);
                        dx = x3-x2;
                        dy = y3-y2;
                        dz = z3-z2;
                        d23 = dx*dx+dy*dy+dz*dz;
                        if (d23<=dmax2){
                            d23 = sqrt(d23);
                            a = (int)(d12*ds);
                            b = (int)(d13*ds);
                            c = (int)(d23*ds);
                            atomicAdd(&XXX[a][b][c],1);
                        }
                    }
                }
                for (k=i+1; k<len2; ++k){
                    // 3er punto en N2
                    x3 = elements2[k].x;
                    y3 = elements2[k].y;
                    z3 = elements2[k].z;
                    dx = x3-x1;
                    dy = y3-y1;
                    dz = z3-z1;
                    d13 = dx*dx+dy*dy+dz*dz;
                    if (d13<=dmax2){
                        d13 = sqrt(d13);
                        dx = x3-x2;
                        dy = y3-y2;
                        dz = z3-z2;
                        d23 = dx*dx+dy*dy+dz*dz;
                        if (d23<=dmax2){
                            d23 = sqrt(d23);
                            a = (int)(d12*ds);
                            b = (int)(d13*ds);
                            c = (int)(d23*ds);
                            atomicAdd(&XXX[a][b][c],1);
                        }
                    }
                }
            }
        }
    }
}

__device__
void count_3_N123(Punto *elements1, unsigned int len1, Punto *elements2, unsigned int len2, Punto *elements3, unsigned int len3, unsigned int ***XXX, float dmax2, float ds, float size_node){
    /*
    Funcion para contar los triangulos en tres 
    nodos con un puntos en N1, un punto en N2
    y un punto en N3.

    row, col, mom => posición de N1.
    u, v, w => posición de N2.
    a, b, c => posición de N3.

    */
    int i,j,k;
    int a,b,c;
    float dx,dy,dz;
    float d12,d13,d23;
    float x1,y1,z1,x2,y2,z2,x3,y3,z3;

    for (i=0; i<len1; ++i){
        // 1er punto en N1
        x1 = elements1[i].x;
        y1 = elements1[i].y;
        z1 = elements1[i].z;
        for (j=0; j<len3; ++j){
            // 2do punto en N3
            x3 = elements3[j].x;
            y3 = elements3[j].y;
            z3 = elements3[j].z;
            dx = x3-x1;
            dy = y3-y1;
            dz = z3-z1;
            d13 = dx*dx+dy*dy+dz*dz;
            if (d13<=dmax2){
                d13 = sqrt(d13);
                for (k=0; k<len2; ++k){
                    // 3er punto en N2
                    x2 = elements2[k].x;
                    y2 = elements2[k].y;
                    z2 = elements2[k].z;
                    dx = x3-x2;
                    dy = y3-y2;
                    dz = z3-z2;
                    d23 = dx*dx+dy*dy+dz*dz;
                    if (d23<=dmax2){
                        d23 = sqrt(d23);
                        dx = x2-x1;
                        dy = y2-y1;
                        dz = z2-z1;
                        d12 = dx*dx+dy*dy+dz*dz;
                        if (d12<=dmax2){
                            d12 = sqrt(d12);
                            a = (int)(d12*ds);
                            b = (int)(d13*ds);
                            c = (int)(d23*ds);
                            atomicAdd(&XXX[a][b][c],1);
                        }
                    }
                }
            }
        }
    }
}

// Kernel function to populate the grid of nodes
__global__
void histo_XXX(Node ***tensor_node, unsigned int ***XXX, unsigned int partitions, float dmax2, float dmax, float ds, float size_node){

    if (blockIdx.x<partitions && threadIdx.x<partitions && threadIdx.y<partitions ){
        // Esto es para el nodo pivote.
        unsigned int row, col, mom;
        row = threadIdx.x;
        col = threadIdx.y;
        mom = blockIdx.x;
        
        int t = 0;

        if (row == 5 && col == 5 && mom == 5){
            t = 1;
        } else {
            t = 0;
        }

        //Contar triangulos dentro del mismo nodo
        count_3_N111(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  XXX, dmax2, ds, t);

        //Para entre nodos
        
        unsigned int u, v, w, a ,b, c; //Indices del nodo 2 (u, v, w) y del nodo 3 (a, b, c)
        unsigned int dis_nod12, dis_nod23, dis_nod31;
        //unsigned int internode_max = (int)(dmax/size_node);
        //unsigned int internode_max2 = (int)(dmax2/(size_node*size_node));
        //float x1N=row, y1N=col, z1N=mom, x2N, y2N, z2N, x3N, y3N, z3N;
        unsigned int dx_nod12, dy_nod12, dz_nod12, dx_nod23, dy_nod23, dz_nod23, dx_nod31, dy_nod31, dz_nod31;

        /*
        //=======================
        // Nodo 2 movil en Z:
        //=======================
        for (dis_nod12=1; dis_nod12<=internode_max && (mom+dis_nod12)<partitions; dis_nod12++){
            w = mom+dis_nod12;
            //==============================================
            // 2 puntos en N y 1 punto en N'
            //==============================================
            count_3_N112(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][col][w].elements, tensor_node[row][col][w].len, XXX, dmax2, ds, size_node);
            
            //==============================================
            // 1 punto en N1, 1 punto en N2 y 1 punto en N3
            //==============================================
            //=======================
            // Nodo 3 movil en Z:
            //=======================
            for (dis_nod31=dis_nod12+1; dis_nod31<=internode_max && (mom+dis_nod31)<partitions; dis_nod31++){ 
                c=mom+dis_nod31;
                count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][col][w].elements, tensor_node[row][col][w].len, tensor_node[row][col][c].elements, tensor_node[row][col][c].len, XXX, dmax2, ds, size_node);
            }

            //=======================
            // Nodo 3 movil en ZY:
            //=======================
            for (dy_nod31=1; dy_nod31<=internode_max && (col+dy_nod31)<partitions; dy_nod31++){
                b = col + dy_nod31;
                for (c = (internode_max<mom)*(mom-internode_max); c<=mom+internode_max && c<partitions; c++ ){
                    dz_nod31=c-mom;
                    dis_nod31 = dy_nod31*dy_nod31 + dz_nod31*dz_nod31;
                    if (dis_nod31 <= internode_max2){
                        dis_nod23 = dy_nod31*dy_nod31 + (c-w)*(c-w); // La dy es la misma entre 3 y 1 que 32 porque el nodo 2 solo se mueve en z (por ahora)
                        if (dis_nod23 <= internode_max2){
                            count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][col][w].elements, tensor_node[row][col][w].len, tensor_node[row][b][c].elements, tensor_node[row][b][c].len, XXX, dmax2, ds, size_node);
                        }
                    }
                }
            }

            //=======================
            // Nodo 3 movil en ZYX:
            //=======================
            for (dx_nod31 = 1; dx_nod31<=internode_max && (row+dx_nod31)<partitions; dx_nod31++){
                a = row+dx_nod31;
                for (b = (internode_max<col)*(col-internode_max); b<=col+internode_max && b<partitions; b++){
                    dy_nod31 = b-col;
                    for (c = (internode_max<mom)*(mom-internode_max); c<=mom+internode_max && c<partitions; c++){
                        dz_nod31 = c-mom;
                        dis_nod31 = dx_nod31*dx_nod31 + dy_nod31*dy_nod31 + dz_nod31*dz_nod31;
                        if (dis_nod31 <= internode_max2){
                            dis_nod23 = dx_nod31*dx_nod31 + dy_nod31*dy_nod31 + (c-w)*(c-w);
                            if (dis_nod23 <= internode_max2){
                                count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][col][w].elements, tensor_node[row][col][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                            }
                        }
                    }
                }
            }
        }

        
        //=======================
        // Nodo 2 movil en ZY:
        //=======================
        //for (v=col+1; v<partitions ; ++v){
        for (dy_nod12=1; (dy_nod12*dy_nod12)<=internode_max2 && (col+dy_nod12)<partitions; dy_nod12++){
            //y2N = v;
            //dy_nod = y2N-y1N;
            v = col+dy_nod12;
            //for (w=0; w<partitions ; ++w){		
            for (dz_nod12=0; (dz_nod12*dz_nod12)<=internode_max2 && (mom+dz_nod12)<partitions; dz_nod12++){
                //z2N = w;
                //dz_nod = z2N-z1N;
                //dis_nod = dy_nod*dy_nod + dz_nod*dz_nod;
                w = mom+dz_nod12;
                dis_nod12 = dy_nod12*dy_nod12+dz_nod12*dz_nod12;
                if (dis_nod12 <= internode_max2){
                    
                    //==============================================
                    // 2 puntos en N y 1 punto en N'
                    //==============================================
                    count_3_N112(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][v][w].elements, tensor_node[row][v][w].len, XXX, dmax2, ds, size_node);

                    //==============================================
                    // 1 punto en N1, 1 punto en N2 y un punto en N3
                    //==============================================
                    //a = u; //=row
                    //b = v; 

                    //=======================
                    // Nodo 3 movil en Z:
                    //=======================
                    //y3N = b;
                    //dy_nod31= y3N-y1N; //y3 = y2 -> dy_nod31 = dy_nod12
                    //for (c=w+1;  c<partitions; ++c){
                    for (dz_nod23=1; (dz_nod23*dz_nod23)<=internode_max2 && (w+dz_nod23)<partitions; dz_nod23++){
                        //z3N = c;
                        //dz_nod31 = z3N-z1N;
                        //dis_nod31 = dy_nod31*dy_nod31 + dz_nod31*dz_nod31;
                        c = w+dz_nod23;
                        dis_nod31 = dy_nod12*dy_nod12 + (dz_nod12+dz_nod23)*(dz_nod12+dz_nod23);
                        if (dis_nod31 <= internode_max2){
                            //dz_nod23 = z3N-z2N;
                            //dis_nod23 = dz_nod23*dz_nod23;
                            count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[row][v][w].elements, tensor_node[row][v][w].len, tensor_node[row][v][c].elements, tensor_node[row][v][c].len, XXX, dmax2, ds, size_node);
                        }
                    }

                    //=======================
                    // Nodo 3 movil en ZY:
                    //=======================	
                    for (b=v+1; b<partitions; ++b){
                    for (dy_nod23; (dy_nod23*dy_nod23)<=internode_max2 && (v+dy_nod23)<partitions; dy_nod23++){
                        y3N = b;
                        dy_nod2 = y3N-y1N;
                        b = v + dy_nod23;
                        for (c=0;  c<partitions; ++c){
                        for (dz_nod31=0; (dz_nod31*dz_nod31)<internode_max2 && ())
                            z3N = c;
                            dz_nod2 = z3N-z1N;
                            //dis_nod2 = dy_nod2*dy_nod2 + dz_nod2*dz_nod2;
                            dis_nod31 = ()*() + ()*();
                            if (dis_nod31 <= internode_max2){
                                dy_nod3 = y3N-y2N;
                                dz_nod3 = z3N-z2N;
                                dis_nod3 = dy_nod3*dy_nod3 + dz_nod3*dz_nod3;
                                if (dis_nod3 <= internode_max2){
                                    count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                                }
                            }
                        }
                    }

                    //=======================
                    // Nodo 3 movil en ZYX:
                    //=======================
                    for (a=u+1; a<partitions; ++a){
                        x3N = a;
                        dx_nod2 = x3N-x1N;
                        for (b=0; b<partitions; ++b){
                            y3N = b;
                            dy_nod2 = y3N-y1N;
                            for (c=0;  c<partitions; ++c){
                                z3N = c;
                                dz_nod2 = z3N-z1N;
                                dis_nod2 = dx_nod2*dx_nod2 + dy_nod2*dy_nod2 + dz_nod2*dz_nod2;
                                if (dis_nod2 <= internode_max2){
                                    dx_nod3 = x3N-x2N;
                                    dy_nod3 = y3N-y2N;
                                    dz_nod3 = z3N-z2N;
                                    dis_nod3 = dx_nod3*dx_nod3 + dy_nod3*dy_nod3 + dz_nod3*dz_nod3;
                                    if (dis_nod3 <= internode_max2){
                                        count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                                    }
                                }
                            }
                        }
                    }
                }
            }	
        }

        
        //=======================
        // Nodo 2 movil en ZYX:
        //=======================
        for (u=row+1; u<partitions; ++u){
            x2N = u;
            dx_nod = x2N-x1N;
            for (v=0; v<partitions; ++v){
                y2N = v;
                dy_nod = y2N-y1N;
                for (w=0; w<partitions; ++w){
                    z2N = w;
                    dz_nod = z2N-z1N;
                    dis_nod = dx_nod*dx_nod + dy_nod*dy_nod + dz_nod*dz_nod;
                    if (dis_nod <= internode_max2){
                        //==============================================
                        // 2 puntos en N y 1 punto en N'
                        //==============================================
                        count_3_N112(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, XXX, dmax2, ds, size_node);
                        //==============================================
                        // 1 punto en N1, 1 punto en N2 y 1 punto en N3
                        //==============================================
                        a = u;
                        b = v;
                        //=======================
                        // Nodo 3 movil en Z:
                        //=======================
                        x3N = a;
                        y3N = b;
                        dx_nod2 = x3N-x1N;
                        dy_nod2 = y3N-y1N;
                        for (c=w+1;  c<partitions; ++c){	
                            z3N = c;
                            dz_nod2 = z3N-z1N;
                            dis_nod2 = dx_nod2*dx_nod2 + dy_nod2*dy_nod2 + dz_nod2*dz_nod2;
                            if (dis_nod2 <= internode_max2){
                                dz_nod3 = z3N-z2N;
                                dis_nod3 = dz_nod3*dz_nod3;
                                if (dis_nod3 <= internode_max2){
                                    count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                                }
                            }
                        }
                            
                        //=======================
                        // Nodo 3 movil en ZY:
                        //=======================
                        for (b=v+1; b<partitions; ++b){
                            y3N = b;
                            dy_nod2 = y3N-y1N;
                            for (c=0;  c<partitions; ++c){
                                z3N = c;
                                dz_nod2 = z3N-z1N;
                                dis_nod2 = dx_nod2*dx_nod2 + dy_nod2*dy_nod2 + dz_nod2*dz_nod2;
                                if (dis_nod2 <= internode_max2){
                                    dy_nod3 = y3N-y2N;
                                    dz_nod3 = z3N-z2N;
                                    dis_nod3 = dy_nod3*dy_nod3 + dz_nod3*dz_nod3;
                                    if (dis_nod3 <= internode_max2){
                                        count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                                    }
                                }
                            }
                        }

                        //=======================
                        // Nodo 3 movil en ZYX:
                        //=======================		
                        for (a=u+1; a<partitions; ++a){
                            x3N = a;
                            dx_nod2 = x3N-x1N;
                            for (b=0; b<partitions; ++b){
                                y3N = b;
                                dy_nod2 = y3N-y1N;
                                for (c=0;  c<partitions; ++c){
                                    z3N = c;
                                    dz_nod2 = z3N-z1N;
                                    dis_nod2 = dx_nod2*dx_nod2 + dy_nod2*dy_nod2 + dz_nod2*dz_nod2;
                                    if (dis_nod2 <= internode_max2){
                                        dx_nod3 = x3N-x2N;
                                        dy_nod3 = y3N-y2N;
                                        dz_nod3 = z3N-z2N;
                                        dis_nod3 = dx_nod3*dx_nod3 + dy_nod3*dy_nod3 + dz_nod3*dz_nod3;
                                        if (dis_nod3 <= internode_max2){
                                            count_3_N123(tensor_node[row][col][mom].elements, tensor_node[row][col][mom].len,  tensor_node[u][v][w].elements, tensor_node[u][v][w].len, tensor_node[a][b][c].elements, tensor_node[a][b][c].len, XXX, dmax2, ds, size_node);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        */

        if (row == 5 && col == 5 && mom == 5){
            printf("Exit the kernel \n");
        }

    }
}

/*
void add_neighbor(int *&array, int &lon, int id){
    lon++;
    int *array_aux;
    cudaMallocManaged(&array_aux, lon*sizeof(int)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i] = array[i];
    }
    cudaFree(&array);
    array = array_aux;
    array[lon-1] = id;
}
*/

//=================================================================== 
void add(Punto *&array, int &lon, float _x, float _y, float _z){
    lon++;
    Punto *array_aux; // = new Punto[lon];
    cudaMallocManaged(&array_aux, lon*sizeof(Punto)); 
    for (int i=0; i<lon-1; i++){
        array_aux[i].x = array[i].x;
        array_aux[i].y = array[i].y;
        array_aux[i].z = array[i].z;
    }
    cudaFree(&array);
    array = array_aux;
    array[lon-1].x = _x;
    array[lon-1].y = _y; 
    array[lon-1].z = _z; 
}

void make_nodos(Node ***nod, Punto *dat, unsigned int partitions, float size_node, unsigned int n_pts){
    /*
    Función para crear los nodos con los datos y puntos random

    Argumentos
    nod: arreglo donde se crean los nodos.
    dat: datos a dividir en nodos.

    */
    int row, col, mom;
    //int node_id, n_row, n_col, n_mom, internodal_distance, id_max = pow((int) dmax/size_node + 1,2); // Row, Col and Mom of the possible node in the neighborhood

    // Inicializamos los nodos vacíos:
    cout << "Initialize empty nodes" << endl;
    for (row=0; row<partitions; row++){
        for (col=0; col<partitions; col++){
            for (mom=0; mom<partitions; mom++){

                nod[row][col][mom].len = 0;
                cudaMallocManaged(&nod[row][col][mom].elements, sizeof(Punto));
            }
        }
    }
    cout << "The nodes have 0 elements each and 0 neighbors" << endl;

    // Llenamos los nodos con los puntos de dat:
    cout << "Started the classification" << endl;
    for (int i=0; i<n_pts; i++){
        row = (int)(dat[i].x/size_node);
        col = (int)(dat[i].y/size_node);
        mom = (int)(dat[i].z/size_node);
        add(nod[row][col][mom].elements, nod[row][col][mom].len, dat[i].x, dat[i].y, dat[i].z);
    }
    cout << "Finished the classification" << endl;
}

//=================================================================== 
void symmetrize(unsigned int ***XXX, unsigned int bn){
    int i,j,k;
    float elem;
    for (i=0; i<bn; i++){
    for (j=i; j<bn; j++){
    for (k=j; k<bn; k++){
        elem = XXX[i][j][k] + XXX[k][i][j] + XXX[j][k][i] + XXX[j][i][k] + XXX[k][j][i] + XXX[i][k][j];
        XXX[i][j][k] = elem;
        XXX[k][i][j] = elem;
        XXX[j][k][i] = elem;
        XXX[j][i][k] = elem;
        XXX[k][j][i] = elem;
        XXX[i][k][j] = elem;
    }   
    }
    }
}

int main(int argc, char **argv){

    string data_loc = argv[1];
    string rand_loc = argv[2];
    string mypathto_files = "../../fake_DATA/DATOS/";
    //This creates the full path to where I have my data files
    data_loc.insert(0,mypathto_files);
    rand_loc.insert(0,mypathto_files);
    
    unsigned int n_pts = stoi(argv[3]), bn=stoi(argv[4]);
    unsigned int n_even = n_pts+(n_pts%2!=0);
    float dmax=stof(argv[5]), size_box = 250.0, size_node = 2.17*size_box/bn;
    float dmax2 = dmax*dmax, ds = ((float)(bn))/dmax;
    unsigned int partitions = (int)(ceil(size_box/size_node));
    
    // Crea los histogramas
    //cout << "Histograms initialization" << endl;
    unsigned int ***DDD;
    // inicializamos los histogramas
    cudaMallocManaged(&DDD, bn*sizeof(unsigned int**));
    for (int i=0; i<bn; i++){
        cudaMallocManaged(&*(DDD+i), bn*sizeof(unsigned int*));
        for (int j = 0; j < bn; j++){
            cudaMallocManaged(&*(*(DDD+i)+j), bn*sizeof(unsigned int));
        }
    }
    //Inicializa en 0
    for (int i=0; i<bn; i++){
        for (int j=0; j<bn; j++){
            for (int k = 0; k < bn; k++){
                DDD[i][j][k]= 0;
            }
        }
    }
    //cout << "Finished histograms initialization" << endl;

    //cout << "Starting to read the data files" << endl;
    Punto *data, *rand; //Crea un array de n_pts puntos
    cudaMallocManaged(&data, n_pts*sizeof(Punto));
    cudaMallocManaged(&rand, n_pts*sizeof(Punto));
    //Llama a una funcion que lee los puntos y los guarda en la memoria asignada a data y rand
    read_file(data_loc,data);
    read_file(rand_loc,rand);
    cout << "Successfully readed the data" << endl;

    //Create Nodes
    //cout << "Started nodes initialization" << endl;
    Node ***nodeD;
    cudaMallocManaged(&nodeD, partitions*sizeof(Node**));
    for (int i=0; i<partitions; i++){
        cudaMallocManaged(&*(nodeD+i), partitions*sizeof(Node*));
        for (int j=0; j<partitions; j++){
            cudaMallocManaged(&*(*(nodeD+i)+j), partitions*sizeof(Node));
        }
    }
    //cout << "Finished nodes initialization" << endl;
    //cout << "Started the data classification into the nodes." << endl;
    make_nodos(nodeD, data, partitions, size_node, n_pts);
    cout << "Finished the data classification in node" << endl;

    //cout << "Calculating the nuber of blocks and threads for the kernel for XXX" << endl;
    //Sets GPU arrange of threads
    int threads=1, blocks=n_even, threads_test, blocks_test;
    float score=pow(blocks,2)+pow((blocks*threads)-n_even,2), score_test;
    for (int i=1; i<6; i++){
        threads_test = pow(2,i);
        blocks_test = (int)(n_even/threads_test)+1;
        score_test = pow(blocks_test,2)+pow((blocks_test*threads_test)-n_even,2);
        
        if (score_test<score){
            threads=threads_test;
            blocks=blocks_test;
            score=score_test;
        }
    }
    
    cout << "Entering to the kernel" << endl;
    clock_t begin = clock();

    cout << partitions << endl;
    cout << DDD[1][2][3] << endl;
    
    dim3 grid(16,1,1);
    dim3 block(16,16);
    histo_XXX<<<grid,block>>>(nodeD, DDD, partitions, dmax2, dmax, ds, size_node);

    //Waits for the GPU to finish
    cudaDeviceSynchronize();

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("\nTiempo en CPU usado = %.4f seg.\n", time_spent );

    //symmetrize(DDD, bn);
    cout << nodeD[1][2][3].elements[1].x << endl;
    cout << nodeD[1][2][3].elements[0].x << endl;
    cout << DDD[1][2][3] << endl;

    cout << DDD[0][0][0] << endl;

    save_histogram("DDD.res", bn, DDD);
    cout << "\nGuarde histograma DDD..." << endl;
    
    // Free memory
    // Free the histogram arrays
    cout << "Free the histograms allocated memory" << endl;
    for (int i=0; i<bn; i++){
        for (int j = 0; j < bn; j++){
            cudaFree(&*(*(DDD+i)+j));
        }
        cudaFree(&*(DDD+i));
    }
    cudaFree(&DDD);
    //Free the nodes and their inner arrays.
    cout << "Free the nodes allocated memory" << endl;
    for (int i=0; i<partitions; i++){
        for (int j=0; j<partitions; j++){
            cudaFree(&*(*(nodeD+i)+j));
        }
        cudaFree(&*(nodeD+i));
    }
    cudaFree(&nodeD);
    //Free data and random arrays
    cout << "Free the data allocated memory" << endl;
    cudaFree(&data);
    cudaFree(&rand);

    cout << "Finished the program" << endl;

    return 0;
}
