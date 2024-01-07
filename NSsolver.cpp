// Need to use actual GNU gcc to compile this code as Apple's
// alias to gcc, clang, does not support OpenMP
// g++-13 -fopenmp -std=c++11 -Wall -Wextra NSsolver.cpp -o NSsolver

#include "NSsolver.hh"
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

// Struct used to keep track of velocity in each cell
struct velStruct{
  double ux;
  double uy;
  double uz;
};

double LL, WW, HH; // Length of container sides, cm
double dx,dt; // Grid spacing (cm), Time stepping (seconds)
double T; // Final time, seconds
int L, W, H; // Number of cells in each direction
int N; // Number of circulation cells
double Ub = 1; // Bubble speed, cm per second
double D = 0.0000185; // Diffusion, cm squared per second
double mu = 1; // viscosity

double *** C; // Gas concentration c in each cell, g per cm^3
double *** P; // Fluid pressure in each cell, Newtons per cm^2 ?
velStruct *** U; // Velocity components in each cell, cm per second 

int MAX_THREADS = 10; // Set for OpenMP, my 2023 M2 Pro has 10 cores, change
                      // when running on other machines with more or less cores

// Used to print out a vertical cross section of the conentration
void printVertical(double *** c){
   int  j = int(W/2);
   for (int k = 1; k < H; k++){
       for (int i = 1; i < L-1; i++){
           printf("%f ", c[i][j][k]);
       }
       printf("\n");
   }
   printf("\n");

}

int main(int argc,char* argv[])
{
    omp_set_num_threads(MAX_THREADS);
    
    if (argc != 6){
        fprintf(stderr, "Usage: ./out <Length> <Width> <Height> <dx> <Final Time> <# Circulation Cells>\n");
        exit(0);
    }

    LL = atof(argv[1]);
    WW = atof(argv[2]);
    HH = atof(argv[3]);
    dx = atof(argv[4]);
    T = atof(argv[5]);
    N = atoi(argv[6]);

    start();
    return 0;
}

void start(){

    // Number of nodes in each direction
    // +2 for ghost nodes on either side
    L = int(LL / dx) + 2;
    W = int(WW / dx) + 2;
    H = int(HH / dx) + 2;
    
    // Allocate and initialize c, P, U
    C = allocate3DArray();
    U = allocate3DVelocity();
    P = allocate3DArray();
    initializeC();
    initializeP();

    enforceBoundary();
    
    // Can test initializations
    //process3DArray(ux);
    //process3DArray(uy);
    //process3DArray(uz);

    // We'll take a time step that's just smaller than the CFL condition
    //double dt = CFL(dx,ux,uy,uz);
    

    int steps = floor(T / dt) + 1;
    fprintf(stderr,"Final time: %f, dt: %f, Images: %d, Length pixels: %d, Width pixels: %d, Height pixels: %d\n", T, dt, steps, L-2, W-2, H-2);
    
    double t_total = 0;
    while (t_total < T){
        step();
        t_total = t_total + dt;
    }
}

void step(){
    return;
}

void enforceBoundary(){
    return;
}

array<double, 3> convert(int i, int j, int k, double dx){
    std::array<double, 3> coord;
    double x = (double(i) - double(L)*.5) * dx + (.5)*dx;
    double y = (double(j) - double(W)*.5) * dx + (.5)*dx;
    double z = (double(k) - (double(H)- 1)) * dx + (.5)*dx;
    coord = {x, y, z};
    
    return coord;
}

bool isBoundary(int i, int j, int k)
{
    return (i == 0 || i == L - 1 || j == 0 || j == W - 1 || k == 0 || k == H - 1);
}

void initializeC(){
    for (int i = 0; i < L; i++){
        for (int j = 0; j < W; j++){
            for (int k = 0; k < H; k++){
                C[i][j][k] = 1; // Dummy
            }
        }
    }
}

void initializeP(){
    for (int i = 0; i < L; i++){
        for (int j = 0; j < W; j++){
            for (int k = 0; k < H; k++){
                P[i][j][k] = 1; // Dummy
            }   
        }   
    }  
}

double ***allocate3DArray(){
    double ***arr = new double **[L];
    for (int i = 0; i < L; i++){
        arr[i] = new double *[W];
        for (int j = 0; j < W; j++){
            arr[i][j] = new double[H];
        }
    }
    return arr;
}

struct velStruct ***allocate3DVelocity(){
    velStruct ***arr = new velStruct **[L];
    for (int i = 0; i < L; i++){        
        arr[i] = new velStruct *[W];
        for(int j = 0; j < W; j++){
            arr[i][j] = new velStruct[H];
        }
    }
    return arr;    
}  

void deallocate3DArray(double ***arr){
    for (int i = 0; i < L; ++i){
        for (int j = 0; j < W; ++j){
            delete[] arr[i][j];
        }
        delete[] arr[i];
    }
    delete[] arr;
}

void process3DArray(double ***arr){
    // Access and manipulate the elements of the 3D array
    for (int k = 0; k < H; k++){
        for (int j = 0; j < W; j++){
            for (int i = 0; i < L; i++){
                std::cout << arr[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
