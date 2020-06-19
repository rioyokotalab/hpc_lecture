// N-S equation demonstration
// High Performance Scitific Computation
// Cavity Lid Driven Flow
// CUDA version
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

// Initialization of variables
void init(double *u, double *un, double *v, double *vn, double *p, double *pn, double *b, int nx, int ny)
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

__global__
void build_up_b(double *b, double *u, double *v, double rho, double dt, double dx, double dy, int nx, int ny)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
//    double term1, term2, term3, term4;
    
    if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
    {
//        term1 = (u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx) +
//        (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy);
//
//        term2 = (u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx);
//
//        term3 = (u[(j+1)*nx+i] - u[(j-1)*nx+i]) *
//        (v[j*nx + i+1] - v[j*nx + i-1]) / (2*2*dx*dy);
//
//        term4 = (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy);
//
//         b[j*nx+i] = (term1) / (dt) - (term2) * (term2) - 2 * (term3) - (term4) * (term4);
        
        b[j*nx+i] =
                   (rho * ( 1.0/dt *
                           ((u[j*nx + i+1] - u[j*nx + i-1]) / (2 * dx)
                          + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -
                   ((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx)) * ((u[j*nx+i+1] - u[j*nx+i-1]) / (2*dx)) -
                   2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2*dy) *
                        (v[j*nx+i+1] - v[j*nx + i-1])   / (2*dx)) -
                   ((v[(j+1)*nx+i] - v[(j-1)*nx+i])     / (2*dy)) * ((v[(j+1)*nx+i] - v[(j-1)*nx+i])     / (2*dy)) ));
    }
    __syncthreads();
}

__global__
void pressure_poisson(double *p, double *pn, double *b, double dx, double dy, int nx, int ny, double rho)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
    
    if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
    {
        p[j*nx+i] =
        (((pn[j*nx+i+1] + pn[j*nx+i-1]) * dy * dy +
           (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) * dx * dx)/
            (2 * (dx * dx + dy * dy)) -
            dx * dx * dy * dy * b[j*nx+i] * rho / (2 * (dx *dx + dy * dy)));
    }
    __syncthreads();
}

__global__
void pressure_boundary(double *p, int nx, int ny)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
    
    if(i == nx - 1 && j > 0 && j < ny - 1)
    {
        p[j*nx+i] = p[j*nx+i-1]; // dp/dx = 0 at x = 2
    }
    
    else if(i == 0 && j > 0 && j < ny - 1)
    {
        p[j*nx+i] = p[j*nx+i+1]; // dp/dx = 0 at x = 0
    }
    
    else if(j == 0 && i < nx )
    {
        p[j*nx+i] = p[(j+1)*nx+i]; // dp/dy = 0 at y = 0
    }
    else if(j == ny - 1 && i < nx)
    {
        p[j*nx+i] = 0.0; // p  = 0 at y = 2
    }
    __syncthreads();
}

__global__
void cavity_flow(double *u, double *un,  double *v, double *vn,  double *p, double *pn, double *b, int nx , int ny, double dx, double dy, double dt, double nu, double rho)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
    
    if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1)
    {
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
    }
    __syncthreads();
}

__global__
void velocity_boundary(double *u, double *v, int nx , int ny)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockDim.y * blockIdx.y ;
    
    if(i == nx - 1 && j > 0 && j < ny - 1)
    {
        u[j*nx+i] = 0.0; // u = 0 at x = 2
        v[j*nx+i] = 0.0; // v = 0 at x = 2
    }
    
    else if(i == 0 && j > 0 && j < ny - 1)
    {
        u[j*nx+i] = 0.0; // u = 0 at x = 0
        v[j*nx+i] = 0.0; // v = 0 at x = 0
    }
    
    else if(j == 0 && i < nx )
    {
        u[j*nx+i] = 0.0; // u = 0 at y = 0
        v[j*nx+i] = 0.0; // v = 0 at y = 0
    }
    else if(j == ny - 1 && i < nx)
    {
        u[j*nx+i] = 1.0; // u = 1.0 at y = 2
        v[j*nx+i] = 0.0; // v = 0 at y = 2
    }
    __syncthreads();
}

// Check of relative error of u and v between adjacent time steps
double Error(double *u, double *un, double *v, double *vn, int nx , int ny)
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
void output_u(int m, double *u, double *v, int nx)
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
    
    double *u, *un, *v, *vn, *p, *pn, *b, error;
    int size = nx * ny * sizeof(double);
    
    u  = (double*)malloc(size);
    un = (double*)malloc(size);
    v  = (double*)malloc(size);
    vn = (double*)malloc(size);
    p  = (double*)malloc(size);
    pn = (double*)malloc(size);
    b  = (double*)malloc(size);
    
    init(u, un, v, vn, p, pn, b, nx, ny);
    
    double *u_g, *un_g, *v_g, *vn_g, *p_g, *pn_g, *b_g;
    cudaMalloc((void**)&u_g , size);
    cudaMalloc((void**)&un_g , size);
    cudaMalloc((void**)&v_g , size);
    cudaMalloc((void**)&vn_g , size);
    cudaMalloc((void**)&p_g , size);
    cudaMalloc((void**)&pn_g , size);
    cudaMalloc((void**)&b_g , size);
    
    cudaMemcpy(u_g,   u,   size, cudaMemcpyHostToDevice);
    cudaMemcpy(un_g,  un,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(v_g,   v,   size, cudaMemcpyHostToDevice);
    cudaMemcpy(vn_g,  vn,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_g,   p,   size, cudaMemcpyHostToDevice);
    cudaMemcpy(pn_g,  pn,  size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_g,   b,   size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock( 128, 1 );
    dim3 blockNumber( (nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    double *p_temp, *u_temp, *v_temp;
    
    for(int loops = 1; loops <= nt ; loops++)
    {

        build_up_b<<<blockNumber,threadsPerBlock>>>(b_g, u_g, v_g, rho, dt, dx, dy, nx ,ny);
        cudaDeviceSynchronize();
        
         for(int k = 0; k < nit ; k++){
             pressure_poisson<<<blockNumber,threadsPerBlock>>>(p_g, pn_g, b_g, dx, dy, nx ,ny, rho);
             pressure_boundary<<<blockNumber,threadsPerBlock>>>(p_g, nx, ny);
            
             p_temp = pn_g; pn_g = p_g; p_g = p_temp;
         }
        cudaDeviceSynchronize();
        
        cavity_flow<<<blockNumber,threadsPerBlock>>>(u_g, un_g, v_g, vn_g, p_g, pn_g, b_g, nx, ny, dx, dy, dt, nu, rho);
        velocity_boundary<<<blockNumber,threadsPerBlock>>>(u_g, v_g, nx, ny);

        u_temp = un_g; un_g = u_g ; u_g = u_temp;
        v_temp = vn_g; vn_g = v_g ; v_g = v_temp;
        cudaDeviceSynchronize();
        
        if (loops % 100 == 0)
        {
            cudaMemcpy(u ,  u_g ,  size, cudaMemcpyDeviceToHost);
            cudaMemcpy(v ,  v_g ,  size, cudaMemcpyDeviceToHost);
            cudaMemcpy(un , un_g,  size, cudaMemcpyDeviceToHost);
            cudaMemcpy(vn , vn_g,  size, cudaMemcpyDeviceToHost);
            cudaMemcpy(p ,  p_g,   size, cudaMemcpyDeviceToHost);
            
            error = Error(u, un, v, vn, nx, ny);
            cout << loops << " loops has been operated." <<endl;
            cout << "The relative error is : " << error <<endl;
//            output_paraview_format(loops, nx, ny, 1, u, v, v, p);
            output_u(loops, u, v, nx);
        }
    }
    cout << "Computation ends." << endl;
    
    return 0;
}

// Lian Tongda
// Jun 19 2020
