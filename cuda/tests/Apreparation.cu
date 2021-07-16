//Criterion
#include <criterion/criterion.h>
//Standard C libraries
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//PCF libraries
#include "../src/create_grid.cuh"
#include "../src/pcf2ani.cuh"

int sample_size=320, bins=0, partitions=35, n_randfiles = 1;
int bpc = 0, analytic=0, rand_dir = 0, rand_required=0, pip_calculation=0; //Used as bools
float size_box_provided = 0, dmax=0;
char *data_name=new char[20], *rand_name=new char[20];

void setup_wdir(void) {
    rand_dir=1;
    rand_required=1;
    dmax=150;
    strcpy(data_name, "data.dat");
    strcpy(rand_name, "rtest/");
    bins=20;
}

void teardown(void) {
    delete[] data_name;
    delete[] rand_name;
}

int cmpflt(float f1, float f2)
{
    float precision = 0.0001;
    if (((f1 - precision) < f2) && ((f1 + precision) > f2))
        return 1;
    else
        return 0;
}

TestSuite(pcf_prep_dirtests,  .init = setup_wdir, .fini = teardown);

Test(pcf_prep_dirtests, read_data_files)
{
    PointW3D *dataD=NULL;
    int np;
    open_files(data_name, &dataD, &np);
    cr_assert(dataD!=NULL, "data_name should not be null\n");
    cr_assert(np==32768, "Should have readed 32768 points readed %i \n", np);
    //First point
    cr_assert_float_eq(dataD[0].x, 1.598884, 0.000001, "Wrong first x got %f should be 1.598884\n", dataD[0].x);
    cr_assert_float_eq(dataD[0].y, 1.722954, 0.000001, "Wrong first y got %f should be 1.722954\n", dataD[0].y);
    cr_assert_float_eq(dataD[0].z, 6.124214, 0.000001, "Wrong first z got %f should be 6.124214\n", dataD[0].z);
    cr_assert_float_eq(dataD[0].w, 1, 0.000001, "Wrong first w got %f should be 1\n", dataD[0].w);
    //Middle point
    cr_assert_float_eq(dataD[412].x, 246.674637, 0.000001, "Wrong first x got %f should be 246.674637\n", dataD[412].x);
    cr_assert_float_eq(dataD[412].y, 90.908241, 0.000001, "Wrong first y got %f should be 90.908241\n", dataD[412].y);
    cr_assert_float_eq(dataD[412].z, 232.052979, 0.000001, "Wrong first z got %f should be 232.052979\n", dataD[412].z);
    cr_assert_float_eq(dataD[412].w, 1, 0.000001, "Wrong first w got %f should be 1\n", dataD[412].w);
    //Last point
    cr_assert_float_eq(dataD[np-1].x, 243.032166, 0.000001, "Wrong first x got %f should be 243.032166\n", dataD[np-1].x);
    cr_assert_float_eq(dataD[np-1].y, 238.387161, 0.000001, "Wrong first y got %f should be 238.387161\n", dataD[np-1].y);
    cr_assert_float_eq(dataD[np-1].z, 233.842285, 0.000001, "Wrong first z got %f should be 233.842285\n", dataD[np-1].z);
    cr_assert_float_eq(dataD[np-1].w, 1, 0.000001, "Wrong first w got %f should be 1\n", dataD[np-1].w);
    free(dataD);

}

Test(pcf_prep_dirtests, read_rand_files)
{
    char **rand_files = NULL, **histo_names = NULL;
    int *rnp = NULL, i;
    PointW3D **dataR=NULL;
    read_random_files(&rand_files, &histo_names, &rnp, &dataR, &n_randfiles, rand_name, rand_dir);
    histo_names[0] = strdup("data.dat");
    cr_assert(n_randfiles==4, "Should have readed 4 files, but readed %i \n", n_randfiles);
    cr_assert_not_null(rand_files, "rand_files is NULL\n");
    cr_assert_not_null(histo_names, "histo_names is NULL\n");
    cr_assert_not_null(rnp, "rnp is NULL\n");
    cr_assert_not_null(dataR, "dataR is NULL\n");
    cr_assert_str_eq(histo_names[0], "data.dat", "The name of the data histogram was not assigned properly \n");
    cr_assert(
        (strcmp(histo_names[1],"rand0_10K.dat")==0 || strcmp(histo_names[2],"rand0_10K.dat")==0 || 
         strcmp(histo_names[3],"rand0_10K.dat")==0 || strcmp(histo_names[4],"rand0_10K.dat")==0),
       "rand0_10K.dat was not in hist_names\n"
    );
    cr_assert(
        (strcmp(histo_names[1],"rand0_1K.dat")==0 || strcmp(histo_names[2],"rand0_1K.dat")==0 || 
         strcmp(histo_names[3],"rand0_1K.dat")==0 || strcmp(histo_names[4],"rand0_1K.dat")==0),
       "rand0_1K.dat was not in hist_names\n"
    );
    cr_assert(
        (strcmp(histo_names[1],"rand0_5K.dat")==0 || strcmp(histo_names[2],"rand0_5K.dat")==0 || 
         strcmp(histo_names[3],"rand0_5K.dat")==0 || strcmp(histo_names[4],"rand0_5K.dat")==0),
       "rand0_5K.dat was not in hist_names\n"
    );
    cr_assert(
        (strcmp(histo_names[1],"rand0_500.dat")==0 || strcmp(histo_names[2],"rand0_500.dat")==0 || 
         strcmp(histo_names[3],"rand0_500.dat")==0 || strcmp(histo_names[4],"rand0_500.dat")==0),
       "rand0_500.dat was not in hist_names\n"
    );

    for (i=0; i<5; i++)
        if (strcmp(histo_names[i],"rand0_5K.dat")==0)
            break;
    i--;
    cr_assert(rnp[i]==5000, "For file %s readed %i points but 5000 were expected\n", histo_names[i+1], rnp[i]);
    // Check first point in rand0_5K.dat
    cr_assert_float_eq(dataR[i][0].x, 125.6088, 0.0001, "Wrong first x got %f should be 125.6088\n", dataR[i][0].x);
    cr_assert_float_eq(dataR[i][0].y, 235.2711, 0.0001, "Wrong first y got %f should be 235.2711\n", dataR[i][0].y);
    cr_assert_float_eq(dataR[i][0].z, 125.9603, 0.0001, "Wrong first z got %f should be 125.9603\n", dataR[i][0].z);
    cr_assert_float_eq(dataR[i][0].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][0].w);
    // Check midle point in rand0_5K.dat
    cr_assert_float_eq(dataR[i][1242].x, 18.5645, 0.0001, "Wrong middle x got %f should be 18.5645\n", dataR[i][1242].x);
    cr_assert_float_eq(dataR[i][1242].y, 193.7719, 0.0001, "Wrong middle y got %f should be 193.7719\n", dataR[i][1242].y);
    cr_assert_float_eq(dataR[i][1242].z, 171.6412, 0.0001, "Wrong middle z got %f should be 171.6412\n", dataR[i][1242].z);
    cr_assert_float_eq(dataR[i][1242].w, 1.00000, 0.000001, "Wrong middle w got %f should be 1.0000\n", dataR[i][1242].w);
    // Check last point in rand0_5K.dat
    cr_assert_float_eq(dataR[i][rnp[i]-1].x, 63.0327, 0.0001, "Wrong first x got %f should be 63.0327\n", dataR[i][rnp[i]-1].x);
    cr_assert_float_eq(dataR[i][rnp[i]-1].y, 99.8218, 0.0001, "Wrong first y got %f should be 99.8218\n", dataR[i][rnp[i]-1].y);
    cr_assert_float_eq(dataR[i][rnp[i]-1].z, 135.2273, 0.0001, "Wrong first z got %f should be 135.2273\n", dataR[i][rnp[i]-1].z);
    cr_assert_float_eq(dataR[i][rnp[i]-1].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][rnp[i]-1].w);
    

    for (i=0; i<5; i++)
        if (strcmp(histo_names[i],"rand0_10K.dat")==0)
            break;
    i--;
    cr_assert(rnp[i]==10000, "For file %s readed %i points but 10000 were expected\n", histo_names[i+1], rnp[i]);
    // Check first point in rand0_10K.dat
    cr_assert_float_eq(dataR[i][0].x, 131.8619, 0.0001, "Wrong first x got %f should be 131.8619\n", dataR[i][0].x);
    cr_assert_float_eq(dataR[i][0].y, 58.7404, 0.0001, "Wrong first y got %f should be 58.7404\n", dataR[i][0].y);
    cr_assert_float_eq(dataR[i][0].z, 61.1886, 0.0001, "Wrong first z got %f should be 61.1886\n", dataR[i][0].z);
    cr_assert_float_eq(dataR[i][0].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][0].w);
    // Check midle point in rand0_10K.dat
    cr_assert_float_eq(dataR[i][1242].x, 66.8554, 0.0001, "Wrong middle x got %f should be 66.8554\n", dataR[i][1242].x);
    cr_assert_float_eq(dataR[i][1242].y, 54.6091, 0.0001, "Wrong middle y got %f should be 54.6091\n", dataR[i][1242].y);
    cr_assert_float_eq(dataR[i][1242].z, 199.3298, 0.0001, "Wrong middle z got %f should be 199.3298\n", dataR[i][1242].z);
    cr_assert_float_eq(dataR[i][1242].w, 1.00000, 0.000001, "Wrong middle w got %f should be 1.0000\n", dataR[i][1242].w);
    // Check last point in rand0_10K.dat
    cr_assert_float_eq(dataR[i][rnp[i]-1].x, 229.1118, 0.0001, "Wrong first x got %f should be 229.1118\n", dataR[i][rnp[i]-1].x);
    cr_assert_float_eq(dataR[i][rnp[i]-1].y, 161.0550, 0.0001, "Wrong first y got %f should be 161.0550\n", dataR[i][rnp[i]-1].y);
    cr_assert_float_eq(dataR[i][rnp[i]-1].z, 121.9876, 0.0001, "Wrong first z got %f should be 121.9876\n", dataR[i][rnp[i]-1].z);
    cr_assert_float_eq(dataR[i][rnp[i]-1].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][rnp[i]-1].w);
    
    for (i=0; i<5; i++)
        if (strcmp(histo_names[i],"rand0_500.dat")==0)
            break;
    i--;
    cr_assert(rnp[i]==500, "For file %s readed %i points but 5000 were expected\n", histo_names[i+1], rnp[i]);
    // Check first point in rand0_500.dat
    cr_assert_float_eq(dataR[i][0].x, 70.5029, 0.0001, "Wrong first x got %f should be 70.5029\n", dataR[i][0].x);
    cr_assert_float_eq(dataR[i][0].y, 39.8920, 0.0001, "Wrong first y got %f should be 39.8920\n", dataR[i][0].y);
    cr_assert_float_eq(dataR[i][0].z, 209.3156, 0.0001, "Wrong first z got %f should be 209.3156\n", dataR[i][0].z);
    cr_assert_float_eq(dataR[i][0].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][0].w);
    // Check midle point in rand0_500.dat
    cr_assert_float_eq(dataR[i][242].x, 26.9605, 0.0001, "Wrong middle x got %f should be 26.9605\n", dataR[i][1242].x);
    cr_assert_float_eq(dataR[i][242].y, 2.9320, 0.0001, "Wrong middle y got %f should be 2.9320\n", dataR[i][1242].y);
    cr_assert_float_eq(dataR[i][242].z, 112.1311, 0.0001, "Wrong middle z got %f should be 112.1311\n", dataR[i][1242].z);
    cr_assert_float_eq(dataR[i][242].w, 1.00000, 0.000001, "Wrong middle w got %f should be 1.0000\n", dataR[i][1242].w);
    // Check last point in rand0_500.dat
    cr_assert_float_eq(dataR[i][rnp[i]-1].x, 46.6021, 0.0001, "Wrong first x got %f should be 46.6021\n", dataR[i][rnp[i]-1].x);
    cr_assert_float_eq(dataR[i][rnp[i]-1].y, 10.1602, 0.0001, "Wrong first y got %f should be 10.1602\n", dataR[i][rnp[i]-1].y);
    cr_assert_float_eq(dataR[i][rnp[i]-1].z, 219.2258, 0.0001, "Wrong first z got %f should be 219.2258\n", dataR[i][rnp[i]-1].z);
    cr_assert_float_eq(dataR[i][rnp[i]-1].w, 1.00000, 0.000001, "Wrong first w got %f should be 1.0000\n", dataR[i][rnp[i]-1].w);

    free(histo_names[0]); 
    for (int i=0; i<n_randfiles; i++){
        free(histo_names[i+1]);
        free(rand_files[i]);
        free(dataR[i]);
    }
    free(histo_names);
    free(rand_files);
    free(rnp);
    free(dataR);

}

Test(pcf_prep_dirtests, read_pip_file)
{
    int np=32768, n_pips;
    int32_t *pipsD=NULL;
    int32_t pip_first[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t pip_middle[] = {2078839517, 1092752850, 767665266, 1253688171, 188418701, 966179054, 205809551, 2016493685, 1793712407, 972990695};
    int32_t pip_last[] = {2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147479551, 2147483647, 2147418111};

    open_pip_files(&pipsD, data_name, np, &n_pips);
    cr_assert_not_null(pipsD, "pipsD is null!!\n");
    cr_assert(n_pips==10, "Should read 10 integers, readed %i \n", n_pips);
    //Check first
    for (int i=0; i<n_pips; i++)
        cr_assert(pipsD[i]==pip_first[i], "Pip[%i] in the first entry should be %i but %i was read\n",i ,pipsD[i],pip_first[i]);
    //Check middle
    for (int i=0; i<n_pips; i++)
        cr_assert(pipsD[(4570)*n_pips + i]==pip_middle[i], "Pip[%i] in the last entry should be %i but %i was read\n",i ,pipsD[(4570)*n_pips + i],pip_middle[i]);
    //Check last
    for (int i=0; i<n_pips; i++)
        cr_assert(pipsD[(np-1)*n_pips + i]==pip_last[i], "Pip[%i] in the last entry should be %i but %i was read\n",i ,pipsD[(np-1)*n_pips + i],pip_last[i]);
    free(pipsD);
}

Test(pcf_prep_dirtests, random_sample_wpips)
{   
    int np, n_pips;
    int i, j, k, shufledinordered=0;
    int32_t *pips_ordered=NULL,  *pips_shuffled=NULL;
    PointW3D *data_ordered=NULL, *data_shuffled=NULL;
    open_files(data_name, &data_ordered, &np);
    open_files(data_name, &data_shuffled, &np);
    open_pip_files(&pips_shuffled, data_name, np, &n_pips);
    open_pip_files(&pips_ordered, data_name, np, &n_pips);
    cr_expect(data_ordered[10].x==data_shuffled[10].x, "The data in position %i has different values before the shuffle\n",10);
    random_sample_wpips(&data_shuffled, &pips_shuffled, np, n_pips, sample_size);
    cr_expect(data_ordered[10].x!=data_shuffled[10].x, "The data in position %i has the same value after the shuffle\n",10);

    for (i = 0; i<sample_size; i++)
    {
        for (j = 0; j<np; j++)
        {
            if (cmpflt(data_shuffled[i].x,data_ordered[j].x) && cmpflt(data_shuffled[i].y,data_ordered[j].y) && cmpflt(data_shuffled[i].z,data_ordered[j].z))
            {
                shufledinordered=1;
                break;
            }
        }
        cr_assert(shufledinordered, "The data in the new pos %i was not found in the original points\n",i);
        cr_assert_float_eq(data_shuffled[i].x, data_ordered[j].x, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", data_shuffled[i].x, data_ordered[j].x);
        cr_assert_float_eq(data_shuffled[i].y, data_ordered[j].y, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", data_shuffled[i].y, data_ordered[j].y);
        cr_assert_float_eq(data_shuffled[i].z, data_ordered[j].z, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", data_shuffled[i].z, data_ordered[j].z);
        cr_assert_float_eq(data_shuffled[i].w, data_ordered[j].w, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", data_shuffled[i].w, data_ordered[j].w);
        for (k = 0; k<n_pips; k++)
        {
            cr_assert(pips_shuffled[i*n_pips + k] == pips_ordered[j*n_pips+k], "PIPs do not match with their points after shuffle\n");
        }
        shufledinordered = 0;
    }

    free(pips_ordered);
    free(pips_shuffled);
    free(data_ordered);
    free(data_shuffled);
}

Test(pcf_prep_dirtests, make_nodes)
{
    int np, nonzero_Dnodes;
    float size_box=0, size_node=0;
    PointW3D *dataD;
    DNode *hnodeD_s = NULL;
    open_files(data_name, &dataD, &np);
    cr_assert(np==32768, "Should have readed 32768 points readed %i \n", np);
    for (int i=0; i<np; i++)
    {
        if (dataD[i].x>size_box) size_box=(int)(dataD[i].x)+1;
        if (dataD[i].y>size_box) size_box=(int)(dataD[i].y)+1;
        if (dataD[i].z>size_box) size_box=(int)(dataD[i].z)+1;
    }
    cr_assert_float_eq(size_box,250.0,0.001,"Got a box of size %f, size 250 was expected\n",size_box);
    size_node = size_box/(float)(partitions);
    cr_assert_float_eq(size_node, 7.14285, 0.0001, "Got a different size_node %f but was expecting 7.14285", size_node);

    nonzero_Dnodes = create_nodes(&hnodeD_s, &dataD, partitions, size_node, np);

    cr_assert_not_null(hnodeD_s, "hnodeD_s is NULL\n");
    cr_assert_not_null(dataD, "dataD is NULL\n");
    for (int i=0; i<nonzero_Dnodes; i++)
    {   
        cr_assert(hnodeD_s[i].len!=0, "Got a zero node with non zero i at i=%i\n", i);
        cr_assert(hnodeD_s[i].end!=0, "Got a zero node with non zero i at i=%i\n", i);
        if (i!=0)
        {
            cr_assert_not(hnodeD_s[i].nodepos.x==0 && hnodeD_s[i].nodepos.y==0 && hnodeD_s[i].nodepos.z==0, "Got a zero node with non zero i at i=%i\n", i);
            cr_assert(hnodeD_s[i].start!=0, "Got a zero node with non zero i at i=%i\n", i);
        }
        for (int pt=hnodeD_s[i].start; pt<hnodeD_s[i].end; pt++)
        {
            cr_assert(
                (dataD[pt].x > hnodeD_s[i].nodepos.x) && (dataD[pt].x < hnodeD_s[i].nodepos.x + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
            cr_assert(
                (dataD[pt].y > hnodeD_s[i].nodepos.y) && (dataD[pt].y < hnodeD_s[i].nodepos.y + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
            cr_assert(
                (dataD[pt].z > hnodeD_s[i].nodepos.z) && (dataD[pt].z < hnodeD_s[i].nodepos.z + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
        }
    }

    free(dataD);
    free(hnodeD_s);
}


Test(pcf_prep_dirtests, make_nodes_wpips)
{
    int np, nonzero_Dnodes, n_pips;
    int i, j, k, shufledinordered=0;
    int32_t *pipsD=NULL, *pips_ordered=NULL;
    float size_box=0, size_node=0;
    PointW3D *dataD, *data_ordered;
    DNode *hnodeD_s = NULL;
    open_files(data_name, &dataD, &np);
    open_files(data_name, &data_ordered, &np);
    open_pip_files(&pipsD, data_name, np, &n_pips);
    open_pip_files(&pips_ordered, data_name, np, &n_pips);
    cr_assert(np==32768, "Should have readed 32768 points readed %i \n", np);
    cr_assert(n_pips==10, "Should have readed 10 pips but readed %i \n", n_pips);
    for (int i=0; i<np; i++)
    {
        if (dataD[i].x>size_box) size_box=(int)(dataD[i].x)+1;
        if (dataD[i].y>size_box) size_box=(int)(dataD[i].y)+1;
        if (dataD[i].z>size_box) size_box=(int)(dataD[i].z)+1;
    }
    cr_assert_float_eq(size_box,250.0,0.001,"Got a box of size %f, size 250 was expected\n",size_box);
    size_node = size_box/(float)(partitions);
    cr_assert_float_eq(size_node, 7.14285, 0.0001, "Got a different size_node %f but was expecting 7.14285", size_node);

    nonzero_Dnodes = create_nodes_wpips(&hnodeD_s, &dataD, &pipsD, n_pips, partitions, size_node, np);

    cr_assert_not_null(hnodeD_s, "hnodeD_s is NULL\n");
    cr_assert_not_null(dataD, "dataD is NULL\n");
    for (int i=0; i<nonzero_Dnodes; i++)
    {   
        cr_assert(hnodeD_s[i].len!=0, "Got a zero node with non zero i at i=%i\n", i);
        cr_assert(hnodeD_s[i].end!=0, "Got a zero node with non zero i at i=%i\n", i);
        if (i!=0)
        {
            cr_assert_not(hnodeD_s[i].nodepos.x==0 && hnodeD_s[i].nodepos.y==0 && hnodeD_s[i].nodepos.z==0, "Got a zero node with non zero i at i=%i\n", i);
            cr_assert(hnodeD_s[i].start!=0, "Got a zero node with non zero i at i=%i\n", i);
        }
        for (int pt=hnodeD_s[i].start; pt<hnodeD_s[i].end; pt++)
        {
            cr_assert(
                (dataD[pt].x > hnodeD_s[i].nodepos.x) && (dataD[pt].x < hnodeD_s[i].nodepos.x + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
            cr_assert(
                (dataD[pt].y > hnodeD_s[i].nodepos.y) && (dataD[pt].y < hnodeD_s[i].nodepos.y + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
            cr_assert(
                (dataD[pt].z > hnodeD_s[i].nodepos.z) && (dataD[pt].z < hnodeD_s[i].nodepos.z + size_node), 
                "Point %f %f %f %f out of range for node with pos %f %f %f\n",
                dataD[pt].x,dataD[pt].y,dataD[pt].z,dataD[pt].w,
                hnodeD_s[i].nodepos.x,hnodeD_s[i].nodepos.y,hnodeD_s[i].nodepos.z
            );
        }
    }

    for (i = 0; i<np; i++)
    {
        for (j = 0; j<np; j++)
        {
            if (cmpflt(dataD[i].x,data_ordered[j].x) && cmpflt(dataD[i].y,data_ordered[j].y) && cmpflt(dataD[i].z,data_ordered[j].z))
            {
                shufledinordered=1;
                break;
            }
        }
        cr_assert(shufledinordered, "The data in the new pos %i was not found in the original points\n",i);
        cr_assert_float_eq(dataD[i].x, data_ordered[j].x, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", dataD[i].x, data_ordered[j].x);
        cr_assert_float_eq(dataD[i].y, data_ordered[j].y, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", dataD[i].y, data_ordered[j].y);
        cr_assert_float_eq(dataD[i].z, data_ordered[j].z, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", dataD[i].z, data_ordered[j].z);
        cr_assert_float_eq(dataD[i].w, data_ordered[j].w, 0.0001, "The point %i in shufled and %i in ordered should have matched\n", dataD[i].w, data_ordered[j].w);
        for (k = 0; k<n_pips; k++)
        {
            cr_assert(pipsD[i*n_pips + k] == pips_ordered[j*n_pips+k], "PIPs do not match with their points after shuffle\n");
        }
        shufledinordered = 0;
    }

    free(dataD);
    free(data_ordered);
    free(pipsD);
    free(hnodeD_s);
}