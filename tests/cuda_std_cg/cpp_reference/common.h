#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Check if device supports cooperative launch
inline bool check_cooperative_launch_support() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  if (!prop.cooperativeLaunch) {
    fprintf(stderr, "Device does not support cooperative launch\n");
    return false;
  }

  return true;
}

// Print device info
inline void print_device_info() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  fprintf(stderr, "Device: %s\n", prop.name);
  fprintf(stderr, "Compute Capability: %d.%d\n", prop.major, prop.minor);
  fprintf(stderr, "Cooperative Launch: %s\n",
          prop.cooperativeLaunch ? "YES" : "NO");
  fprintf(stderr, "Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
  fprintf(stderr, "Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0],
          prop.maxGridSize[1], prop.maxGridSize[2]);
}

#endif // COMMON_H
