/** 
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

#include <xmmintrin.h>

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// includes, CUDA
#include <builtin_types.h>

// Complex data type
typedef float2 Complex;

// Int16 to Float32
__global__ void Int16toFloat32(int16_t* a, Complex* b, int batch, long long size);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
//#define SIGNAL_SIZE 50
//#define SIGNAL_SIZE 1024
//#define SIGNAL_SIZE 8192
//#define SIGNAL_SIZE 256
#define SIGNAL_SIZE 0x400000
#define BATCH       8
#define DEVS        2

Complex h_signal[DEVS][SIGNAL_SIZE * BATCH];
Complex h_convolved_signal[DEVS][SIGNAL_SIZE * BATCH];

int16_t h_convolved_signal16[DEVS][SIGNAL_SIZE * 2 * BATCH];


double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed!\n");

    PCFreq = (double)(li.QuadPart) / 1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart - CounterStart) / PCFreq;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) {
    printf("[simpleCUFFT] is starting...\n");

    findCudaDevice(argc, (const char**)argv);



    // Allocate host memory for the signal
  //  Complex *h_signal =
  //      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

    int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE * BATCH;

    int16_t* h_signal16[DEVS];
    for (int j = 0; j < DEVS; j++) {
        checkCudaErrors(cudaMallocHost((void**)&h_signal16[j], mem_size / 2)); // host pinned
    }

  // Initialize the memory for the signal
    for (int j = 0; j < DEVS; j++) {
        for (unsigned int i = 0; i < SIGNAL_SIZE * 2 * BATCH; i += 2) {
            h_signal16[j][i] = 0;
            h_signal16[j][i + 1] = 0;
        }
    }
  
    for (int j = 0; j < DEVS; j++) {
        for (unsigned int i = 0; i < 2 * BATCH; i += 2) {
            h_signal16[j][i] = j*BATCH + i/2 + 1;
            h_signal16[j][i + 1] = 0;
        }
    }

    StartCounter();
    double copytime = GetCounter();

    // Allocate device memory for signal
    int16_t* d_signal_i16[DEVS];
    Complex *d_signal[DEVS];
    Complex* r_signal[DEVS];
    for (int j = 0; j < DEVS; j++) {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal_i16[j]), mem_size / 2));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal[j]), mem_size));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&r_signal[j]), mem_size));
    }
      
  StartCounter();
  // Copy host memory to device
  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cudaMemcpy(d_signal_i16[j], h_signal16[j], mem_size / 2, cudaMemcpyHostToDevice));
  }
  copytime = GetCounter();
  printf("---- Copy time %ld Bytes is: %0.3f milliseconds \n", DEVS*mem_size/2, copytime);
  printf("--------------------------------------------- \n");

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (SIGNAL_SIZE + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  
  StartCounter();
  for (int j = 0; j < DEVS; j++) {
      Int16toFloat32 << <blocksPerGrid, threadsPerBlock >> > (d_signal_i16[j], d_signal[j], BATCH, SIGNAL_SIZE);
  }
  copytime = GetCounter();
  printf("---- Int16toFloat32 kernel time is: %0.3f milliseconds \n", copytime);
  printf("--------------------------------------------- \n");
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch Int16toFloat32 kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cudaMemcpy(h_signal[j], d_signal[j], mem_size, cudaMemcpyDeviceToHost));
  }

  // CUFFT plan simple API
  cufftHandle plan;
  //checkCudaErrors(cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, BATCH));

  checkCudaErrors(cufftCreate(&plan));
  int rank = 1;
  long long int n = SIGNAL_SIZE;
  long long int inembed[] = { 0 };
  long long int istride = 1;// BATCH;
  long long int idist = n;// 1;
  cudaDataType inputtype = CUDA_C_32F;
  long long int onembed[] = { 0 };
  long long int ostride = 1;
  long long int odist = n;
  cudaDataType outputtype = CUDA_C_32F;
  long long int batch = BATCH;
  size_t workSize;
  cudaDataType executiontype = CUDA_C_32F;

  checkCudaErrors(cufftXtMakePlanMany(plan, rank, &n,
      inembed, istride, idist, inputtype,
      onembed, ostride, odist, outputtype,
      batch, &workSize,
      executiontype));

  // Transform signal and kernel
  printf("Transforming signal cufftXtExec\n");
  
  // timer init
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;
  cudaEventCreate(&start, 0);
  cudaEventCreate(&stop, 0);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  StartCounter();

  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cufftXtExec(plan, d_signal[j], r_signal[j], CUFFT_FORWARD));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  double ttime = GetCounter();

  cudaEventElapsedTime(&gpuTime, start, stop);
  printf("---- time: %.10f milliseconds\n", gpuTime);
  printf("---- Execution time is: %0.3f milliseconds \n", ttime);
  printf("--------------------------------------------- \n");

  // Copy device memory to host
  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cudaMemcpy(h_convolved_signal[j], r_signal[j], mem_size, cudaMemcpyDeviceToHost));
  }

  for (int j = 0; j < DEVS; j++) {
      for (int i = 0; i < BATCH; i++) {
          printf("DEVS %d|BATCH %d: %f - %f\n", j, i, h_convolved_signal[j][SIGNAL_SIZE * i].x, h_convolved_signal[j][SIGNAL_SIZE * i].y);
      }
  }

  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cudaMemcpy(h_signal[j], d_signal[j], mem_size, cudaMemcpyDeviceToHost));
  }

  // Check if kernel execution generated and error
   getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

/*
  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftXtExec(plan, r_signal, d_signal, CUFFT_INVERSE));

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                             cudaMemcpyDeviceToHost));
*/
  
  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  for (int j = 0; j < DEVS; j++) {
      checkCudaErrors(cudaFree(d_signal[j]));
      checkCudaErrors(cudaFree(r_signal[j]));
  }

  exit(EXIT_SUCCESS);
}


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////
/*
// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
*/

// Int16 to Float32
__global__ void Int16toFloat32(int16_t *a, Complex *b, int batch, long long size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for(int j=0;j<batch;j++) {
        for (int i = threadID; i < size; i += numThreads) {
            b[i + size * j].x = a[i * 2 * batch + j * 2];
            b[i + size * j].y = a[i * 2 * batch + 1 + j * 2];
        }
    }
}

