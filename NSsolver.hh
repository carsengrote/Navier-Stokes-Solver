#ifndef NSSOLVER_HH
#define NSSOLVER_HH

#include <array>

// i,j,k are x,y,z direction index
// L,W,H are x,y,z direction array length

void start(double LL, double WW, double HH, double dx, double T);
std::array<double, 3> convert(int i, int j, int k, double dx);

double initializeC(int i, int j, int k);
double initializeP(int i, int j, int k);
double initializeUx(int i, int j, int k, double LL, double HH);
double initializeUy(int i, int j, int k, double LL, double HH);
double initializeUz(int i, int j, int k, double LL, double HH);

double CFL(double dx, double ***ux, double ***uy, double ***uz);

bool isBoundary(int i, int j, int k); // given index and bound
void enforceBoundary (double ***c);

double getC(int i, int j, int k);
std::array<double, 3> getU(int i, int j, int k); // return ux, uy, uz at one point

struct ptrStruct updateC(double *** c, double ***cPrime, double ***cPrimeLast, double *** cNext, double *** ux, double *** uy, double *** uz, double dx, double dt, int E);

double *** allocate3DArray();
double ** allocate2DArray();
void deallocate3DArray(double ***arr);
void process3DArray(double ***arr);
void computeC(double ***c, double dx);
void computeUx(double ***ux, double dx, double LL, double HH);
void computeUy(double ***uy, double dx, double LL, double HH);
void computeUz(double ***uz, double dx, double LL, double HH);

#endif
