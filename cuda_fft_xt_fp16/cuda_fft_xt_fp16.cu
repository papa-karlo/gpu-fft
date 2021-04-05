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

#include <cuda_fp16.h>


// Complex data type
typedef half2 halfComplex;
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *,
                                                   int, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

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

halfComplex h_signal_half[SIGNAL_SIZE * BATCH];
halfComplex h_convolved_signal_half[SIGNAL_SIZE * BATCH];
//Complex h_signal[SIGNAL_SIZE * BATCH];
//Complex h_convolved_signal[SIGNAL_SIZE * BATCH];

int16_t h_signal16[SIGNAL_SIZE*2*BATCH];
int16_t h_convolved_signal16[SIGNAL_SIZE*2 * BATCH];

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

  int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE * BATCH;
  int mem_size_half = sizeof(cufftComplex) * (SIGNAL_SIZE/2) * BATCH;

    // host arrays
  Complex *h_PinnedSignal, *h_PinnedConvolvedSignal;
  halfComplex* h_PinnedSignal_half, * h_PinnedConvolvedSignal_half;
  // allocate and initialize
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedSignal, mem_size)); // host pinned
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedConvolvedSignal, mem_size)); // host pinned
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedSignal_half, mem_size_half)); // host pinned
  checkCudaErrors(cudaMallocHost((void**)&h_PinnedConvolvedSignal_half, mem_size_half)); // host pinned


  // Initialize the memory for the signal
  halfComplex* p_half;

  p_half = (halfComplex*)h_signal16;

  for (unsigned int i = 0; i < SIGNAL_SIZE*2*BATCH; i+=2) {
      h_signal16[i] = 0;
      h_signal16[i+1] = 0;
  }

  for (unsigned int i = 0; i < BATCH * 2; i += 2) {
      h_signal16[i] = i + 1;
      h_signal16[i + 1] = 0;
  }
  /*
  for (unsigned int i = 0; i < SIGNAL_SIZE*BATCH; i++) {
      h_signal[i].x = p_half[i].x;
      h_signal[i].y = p_half[i].y;
  }
  */

  StartCounter();
  // Initialize the memory for the signal
#pragma loop(hint_parallel(0))
  for (unsigned int i = 0; i < SIGNAL_SIZE*BATCH; ++i) {
    //h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
    //h_signal[i].y = 0;
//    h_signal_half[i].x = (half)i;
  //  h_signal_half[i].y = (half)0;
    //h_signal[i].x = h_signal_half[i].x;
    //h_signal[i].y = h_signal_half[i].y;
    h_PinnedSignal_half[i].x = h_signal16[2*i];
    h_PinnedSignal_half[i].y = h_signal16[2*i+1];
  }
  
  float xx = h_PinnedSignal_half[0].x;
  xx/=32768;
  h_PinnedSignal_half[0].x = xx;
  float yy = h_PinnedSignal_half[0].x;

  double copytime = GetCounter();
  printf("----- Int2float time %ld Bytes is: %0.3f milliseconds \n", mem_size, copytime);
  printf("--------------------------------------------- \n");
  
  
  // Allocate device memory for signal
  halfComplex *d_signal;
  halfComplex* r_signal;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size_half));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&r_signal), mem_size_half));
  
  StartCounter();
  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_signal, h_signal_half, mem_size_half, cudaMemcpyHostToDevice));
  copytime = GetCounter();
  printf("---- Copy time %ld Bytes is: %0.3f milliseconds \n", mem_size_half, copytime);
  printf("--------------------------------------------- \n");

  StartCounter();
  // Copy host memory to device
  //checkCudaErrors(cudaMemcpy(d_signal, h_PinnedSignal_half, mem_size_half, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_signal, p_half, mem_size_half, cudaMemcpyHostToDevice));
  copytime = GetCounter();
  printf("---- Pinned Memory Copy time %ld Bytes is: %0.3f milliseconds \n", mem_size, copytime);
  printf("--------------------------------------------- \n");

    // CUFFT plan simple API
  cufftHandle plan;
  //checkCudaErrors(cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1));
  checkCudaErrors(cufftCreate(&plan));
  int rank = 1;
  //long long int n = 65536*8*2*4;// 0x1000000;// SIGNAL_SIZE;
 /*
  long long int n = SIGNAL_SIZE/2;
  long long int inembed[] = { 0 };
  long long int istride = 1; 
  long long int idist = n;
  cudaDataType inputtype = CUDA_C_16F;
  long long int onembed[] = { 0 };
  long long int ostride = 1;
  long long int odist = n;
  cudaDataType outputtype = CUDA_C_16F;
*/
  long long int n = SIGNAL_SIZE / 2;
  long long int inembed[] = { 0 };
  long long int istride = 2;
  long long int idist = 1;
  cudaDataType inputtype = CUDA_C_16F;
  long long int onembed[] = { 0 };
  long long int ostride = 1;
  long long int odist = n;
  cudaDataType outputtype = CUDA_C_16F;

  long long int batch = BATCH;// 1;
  size_t workSize;
  cudaDataType executiontype = CUDA_C_16F;
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

  checkCudaErrors(cufftXtExec(plan, d_signal, r_signal, CUFFT_FORWARD));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  double ttime = GetCounter();

  cudaEventElapsedTime(&gpuTime, start, stop);
  printf("---- time: %.10f milliseconds\n", gpuTime);
  printf("---- Execution time is: %0.3f milliseconds \n", ttime);
  printf("--------------------------------------------- \n");

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_convolved_signal_half, r_signal, mem_size_half,
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_signal_half, d_signal, mem_size_half,
      cudaMemcpyDeviceToHost));

  for (int ii = 0; ii < SIGNAL_SIZE; ii++) {
      h_convolved_signal[ii].x = h_convolved_signal_half[ii].x;
      h_convolved_signal[ii].y = h_convolved_signal_half[ii].y;
  }

  // Check if kernel execution generated and error
   getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftXtExec(plan, r_signal, d_signal, CUFFT_INVERSE));

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_convolved_signal_half, d_signal, mem_size_half,
                             cudaMemcpyDeviceToHost));

  
  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  checkCudaErrors(cudaFree(d_signal));
  checkCudaErrors(cudaFree(r_signal));

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