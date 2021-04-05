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
typedef float2  Complex;
typedef double2 DoubleComplex;

// Int16 to Float32
__global__ void Int16toFloat32(int16_t* a, Complex* b, int batch, long long size);

// Int16 to Float64
__global__ void Int16toFloat64(int16_t* a, DoubleComplex* b, int batch, long long size);


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
//#define SIGNAL_SIZE 50
//#define SIGNAL_SIZE 1024
//#define SIGNAL_SIZE 8192
//#define SIGNAL_SIZE 256
#define SIGNAL_SIZE 0x400000
#define BATCH       16

DoubleComplex h_signal[SIGNAL_SIZE * BATCH];
//DoubleComplex h_convolved_signal[SIGNAL_SIZE * BATCH];

//int16_t h_signal16[SIGNAL_SIZE * 2 * BATCH];
int16_t h_convolved_signal16[SIGNAL_SIZE * 2 * BATCH];


double copytime;
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
void runTest(int argc, char **argv) {
  printf("[simpleCUFFT] is starting...\n");

  findCudaDevice(argc, (const char **)argv);

    // Allocate host memory for the signal
//  Complex *h_signal =
//      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));

  int mem_size = sizeof(cufftDoubleComplex) * SIGNAL_SIZE * BATCH;

    // host arrays
  DoubleComplex *h_PinnedSignal, *h_PinnedConvolvedSignal;
  // allocate and initialize
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedSignal, mem_size)); // host pinned
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedConvolvedSignal, mem_size)); // host pinned

  int16_t *h_signal16;
  checkCudaErrors(cudaMallocHost((void**)&h_signal16, mem_size/4)); // host pinned

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE * 2 * BATCH; i+=2) {
      h_signal16[i] = 0;
      h_signal16[i+1] = 0;
  }
  
  for (unsigned int i = 0; i < 2 * BATCH; i += 2) {
      h_signal16[i] = i+1;
      h_signal16[i + 1] = 0;
  }

    // Allocate device memory for signal
  int16_t* d_signal_i16;
  DoubleComplex *d_signal;
  DoubleComplex *r_signal;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal_i16), mem_size/4));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&r_signal), mem_size));
  
  StartCounter();
  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_signal_i16, h_signal16, mem_size/4, cudaMemcpyHostToDevice));
  copytime = GetCounter();
  printf("---- Copy time %ld Bytes is: %0.3f milliseconds \n", mem_size/4, copytime);
  printf("--------------------------------------------- \n");

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (SIGNAL_SIZE + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  
  StartCounter();
  Int16toFloat64 << <blocksPerGrid, threadsPerBlock >> > (d_signal_i16, d_signal, BATCH, SIGNAL_SIZE);
  copytime = GetCounter();
  printf("---- Int16toFloat64 kernel time is: %0.3f milliseconds \n", copytime);
  printf("--------------------------------------------- \n");
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch Int16toFloat32 kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
      cudaMemcpyDeviceToHost));

  // CUFFT plan simple API
  cufftHandle plan;
  checkCudaErrors(cufftCreate(&plan));
  int rank = 1;
  long long int n = SIGNAL_SIZE;
  long long int inembed[] = { 0 };
  long long int istride = 1;// BATCH;
  long long int idist = n;// 1;
  cudaDataType inputtype = CUDA_C_64F;
  long long int onembed[] = { 0 };
  long long int ostride = 1;
  long long int odist = n;
  cudaDataType outputtype = CUDA_C_64F;
  long long int batch = BATCH;
  size_t workSize;
  cudaDataType executiontype = CUDA_C_64F;

  checkCudaErrors(cufftXtMakePlanMany(plan, rank, &n,
      inembed, istride, idist, inputtype,
      onembed, ostride, odist, outputtype,
      batch, &workSize,
      executiontype));

  // Transform signal and kernel
  printf("Transforming signal cufftXtExec\n");
  
  // Launch the Vector Add CUDA Kernel
  threadsPerBlock = 1024;
  blocksPerGrid = (SIGNAL_SIZE + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  cycle:
  // timer init
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;
  cudaEventCreate(&start, 0);
  cudaEventCreate(&stop, 0);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  StartCounter();

  checkCudaErrors(cufftXtExec(plan, d_signal, r_signal, CUFFT_FORWARD));

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  double ttime = GetCounter();

  cudaEventElapsedTime(&gpuTime, start, stop);
  printf("---- time: %.10f milliseconds\n", gpuTime);
  printf("---- Execution time is: %0.3f milliseconds \n", ttime);
  printf("--------------------------------------------- \n");

  goto cycle;

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_signal, r_signal, mem_size,
        cudaMemcpyDeviceToHost));

  for (int i = 0; i < BATCH; i++) {
      printf("BATCH %d: %f : %f\n", i, h_signal[SIGNAL_SIZE * i].x, h_signal[SIGNAL_SIZE * i].y);
  }
  
  checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
      cudaMemcpyDeviceToHost));

  // Check if kernel execution generated and error
   getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
    
  
  
  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  checkCudaErrors(cudaFree(d_signal));
  checkCudaErrors(cudaFree(r_signal));

  exit(EXIT_SUCCESS);
}


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

// Int16 to Float64
__global__ void Int16toFloat64(int16_t* a, DoubleComplex* b, int batch, long long size)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < batch; j++) {
        for (int i = threadID; i < size; i += numThreads) {
            b[i + size * j].x = a[i * 2 * batch + j * 2];
            b[i + size * j].y = a[i * 2 * batch + 1 + j * 2];
        }
    }
}

