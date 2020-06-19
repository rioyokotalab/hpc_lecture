// N-S equation demonstration
// High Performance Scitific Computation
// Cavity Lid Driven Flow
// C++ version
// 19M18085 Lian Tongda

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;

const int nx = 41; // Mesh points in X direction
const int ny = 41; // Mesh points in Y direction
const int nt = 10000;  //Total time steps
const int nit = 50;    // Poisson equation iterations

//const double c = 1;
const double dx = 2.0 / (nx - 1);  // LX = 2.0 , dx = 0.05
const double dy = 2.0 / (ny - 1);  // LY = 2.0 , dy = 0.05

// Re = U_lid * L / nu = 1.0 * 2.0 / 0.1 = 20.0

const double rho = 1.0;     // rho is considered constant in this case
const double nu = 0.1;      // Kinematic viscosity
const double dt = 0.001;    // length of time step

double sq(double a)         // User-defined square function
{
    return (a*a);
}

// Initialization of variables
void init(double *u, double *un, double *v, double *vn, double *p, double *pn, double *b)
{
    int i, j;
    for(i = 0; i < nx ; i++){
        for(j = 0; j < ny ; j++){
            u[j*nx+i] = 0.0;
            un[j*nx+i] = 0.0;
            v[j*nx+i] = 0.0;
            vn[j*nx+i] = 0.0;
            p[j*nx+i] = 0.0;
            pn[j*nx+i] = 0.0;
            b[j*nx+i] = 0.0;
        }
    }
    //    Velocity of Lid
    for(i = 0 ; i < nx ;i++)
    {
        u[(nx-1)*nx + i] = 1.0;
        un[(nx-1)*nx + i] = 1.0;
    }
    
    cout<<"DATA INITIALIZED"<<endl;
}

void build_up_b(double *b, double *u, double *v)
{
    int i, j;
    double term1, term2, term3, term4;
    for(i = 1; i < nx - 1 ; i++){
        for(j = 1; j < ny - 1 ; j++){
            
            b[j*nx+i] =
            (rho * ( 1.0/dt *
                    ((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx)
                     + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
                    ((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx)) * ((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx)) -
                    2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2*dy) *
                         (v[j*nx+i+1] - v[j*nx + i-1])   / (2*dx)) -
                    ((v[(j+1)*nx+i] - v[(j-1)*nx+i])     / (2*dy)) * ((v[(j+1)*nx+i] - v[(j-1)*nx+i])     / (2*dy)) ));
            
        }
    }
    
}

void pressure_poisson(double *p, double *pn, double *b)
{
    
    int i, j;
    //    nit is the number of iteration of Poisson equation
    for(int k = 0; k < nit ; k++){
        
        for(i = 0; i < nx ; i++){
            for(j = 0; j < ny ; j++){
                pn[j*nx+i] = p[j*nx+i];
            }
        }
        
        for(i = 1; i < nx - 1 ; i++){
            for(j = 1; j < ny - 1; j++){
                p[j*nx+i] =
                (((pn[j*nx+i+1] + pn[j*nx+i-1]) * dy * dy +
                  (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) * dx * dx)/
                 (2 * (dx * dx + dy * dy)) -
                 dx * dx * dy * dy * b[j*nx+i] * rho / (2 * (dx *dx + dy * dy)));
            }
        }
        
        for(int m = 1 ; m < ny - 1 ; m++)
        {
            p[m*nx+nx-1] = p[m*nx+nx-2]; // dp/dx = 0 at x = 2
            p[m*nx+0]    = p[m*nx+1];     // dp/dx = 0 at x = 0
        }
        
        for(int n = 0 ; n < nx ; n++)
        {
            p[0*nx+n] = p[1*nx+n]; // dp/dy = 0 at y = 0
            p[(nx-1)*nx+n] = 0.0;  // p  = 0 at y = 2
        }
    }
}

void cavity_flow(double *u, double *un,  double *v, double *vn,  double *p, double *pn, double *b)
{
    int i, j;
    //    buffer of u_old and v_old
    for(i = 0; i < nx ; i++){
        for(j = 0; j < ny ; j++){
            un[j*nx+i] = u[j*nx+i];
            vn[j*nx+i] = v[j*nx+i];
        }
    }
    
    build_up_b(b, u, v);
    pressure_poisson(p, pn, b);
    
    for(i = 1; i < nx - 1 ; i++){
        for(j = 1; j < ny - 1 ; j++){
            u[j*nx+i] = (
                         un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i-1]) -
                         vn[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j-1)*nx+i]) -
                         dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1]) +
                         nu * (dt / (dx * dx) * (un[j*nx+i+1] - 2*un[j*nx+i] + un[j*nx+i-1]) +
                               dt / (dy * dy) * (un[(j+1)*nx+i] - 2*un[j*nx+i] + un[(j-1)*nx+i])));
            
            v[j*nx+i] = (
                         vn[j*nx+i] - un[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i-1]) -
                         vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j-1)*nx+i]) -
                         dt / (2 * rho * dy) * (p[(j+1)*nx+i] - p[(j-1)*nx+i]) +
                         nu * (dt / (dx * dx) * (vn[j*nx+i+1] - 2*vn[j*nx+i] + vn[j*nx+i-1]) +
                               dt / (dy * dy) * (vn[(j+1)*nx+i] - 2*vn[j*nx+i] + vn[(j-1)*nx+i])));
        }}
    
    //    Boundary condition of top and bottom wall
    for(i = 0 ; i < nx ; i++)
    {
        u[0*nx+i] = 0.0;
        u[(nx-1)*nx+i] = 1.0;
        v[0*nx+i] = 0.0;
        v[(nx-1)*nx+i] = 0.0;
    }
    //  Boundary condition of left and right wall
    for(j = 1; j < ny - 1 ; j++)
    {
        u[j*nx+0] = 0.0;
        u[j*nx+nx-1] = 0.0;
        v[j*nx+0] = 0.0;
        v[j*nx+nx-1] = 0.0;
    }
}

// Check of relative error of u and v between adjacent time steps
double Error(double *u, double *un, double *v, double *vn)
{
    double temp1 = 0.0, temp2 = 0.0;
    int i, j, index;
    for(i = 0; i < nx ; i++){
        for(j = 0; j < ny ; j++){
            index =  j* nx +i;
            
            temp1 += (u[index] - un[index]) * (u[index] - un[index])
            + (v[index] - vn[index]) * (v[index] - vn[index]);
            
            temp2 += (u[index] * u[index]) + (v[index] * v[index]);
        }
    }
    temp1 = sqrt(temp1);
    temp2 = sqrt(temp2);
    return ( temp1 / (temp2 + 1e-30));
}

// Output u  and v along the central line.
void output_u(int m, double *u, double *v)
{
    ostringstream name;
    name << "u_" << m << ".txt";
    ofstream out(name.str().c_str());
    
    for(int i = 0; i < nx ; i++){
        out <<  u[i * nx + (nx+1)/2] <<" "<<  v[ nx * (nx + 1) / 2 + i] << endl;
    }
    out.close();
}

int main(void)
{
    double *u, *un, *v, *vn, *p, *pn, *b, error;
    int size = nx * ny * sizeof(double);
    
    u  = (double*)malloc(size);
    un = (double*)malloc(size);
    v  = (double*)malloc(size);
    vn = (double*)malloc(size);
    p  = (double*)malloc(size);
    pn = (double*)malloc(size);
    b  = (double*)malloc(size);
    
    init(u, un, v, vn, p, pn, b);
    
    for(int loops = 1; loops <= nt ; loops++)
    {
        cavity_flow(u, un, v, vn, p, pn ,b);
        
        if (loops % 100 == 0)
        {
            error = Error(u, un, v, vn);
            cout << loops << " loops has been operated." <<endl;
            cout << "The relative error is : " << error <<endl;
            output_u(loops, u, v);
        }
    }
    cout << "Computation ends." << endl;
    return 0;
}

// Jun 19 2020
// Lian Tongda
