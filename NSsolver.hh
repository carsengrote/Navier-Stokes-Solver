#ifndef NSSOLVER_HH
#define NSSOLVER_HH

#include <array>

// i,j,k are x,y,z direction index
// L,W,H are x,y,z direction array length

void start();
void step();
std::array<double, 3> convert(int i, int j, int k, double dx);
void initializeC();
void initializeP();
double CFL();
bool isBoundary(int i, int j, int k); // given index and bound
void enforceBoundary();
double *** allocate3DArray();
double ** allocate2DArray();
struct velStruct ***allocate3DVelocity();
void deallocate3DArray(double ***arr);
void process3DArray(double ***arr);

#endif
