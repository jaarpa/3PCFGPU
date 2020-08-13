#include<iostream>
#include<fstream>
#include<cmath>
#include <vector>
#include <iomanip>

using namespace std;

int main(int argc, const char** argv) {
    float inputData;
    double data_arr[500][3];
    double rand_arr[500][3];
    vector<float> data_v;
    string myText;

    ifstream data("/home/jaarpa/aProjects/DESI/fake_DATA/DATOS/data_500.dat");
    ifstream rand("/home/jaarpa/aProjects/DESI/fake_DATA/DATOS/rand0_500.dat");

    // Use a while loop together with the getline() function to read the file line by line
    while (getline (data, myText)) {
        // Output the text from the file
        cout << myText << endl;
    }

    /*
    while(data>>inputData){ //Lee un dato a la vez
        data_v.push_back(inputData); // Adiere cada elemento al vector
    }
        cout << data_v.size();
    */

    data.close();
    rand.close();

    return 0;
}

