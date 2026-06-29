#ifndef SPATIAL_OPS_CU
#define SPATIAL_OPS_CU

#ifndef __CUDACC__
// Shim definitions to prevent IDE linting errors when not compiling with nvcc
#define __global__
#define __device__
#define __host__
#define __shared__
struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x, y, z;
};
extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;
inline int atomicAdd(int *address, int val) { return 0; }
inline float atomicAdd(float *address, float val) { return 0.0f; }
template <typename T> inline T min(T a, T b) { return a < b ? a : b; }
inline void __syncthreads() {}
#else
#include <cuda_runtime.h>
#endif

extern "C" {

__global__ void kinetic_join_strided_kernel(const float *__restrict__ positions,
                                            int num_robots, float radius,
                                            int *__restrict__ collision_count,
                                            int *__restrict__ collisions,
                                            int max_collisions) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float r2 = radius * radius;

  for (int i = index; i < num_robots; i += stride) {
    float xi = positions[3 * i + 0];
    float yi = positions[3 * i + 1];
    float zi = positions[3 * i + 2];

    for (int j = i + 1; j < num_robots; ++j) {
      float dx = xi - positions[3 * j + 0];
      float dy = yi - positions[3 * j + 1];
      float dz = zi - positions[3 * j + 2];
      float dist_sq = dx * dx + dy * dy + dz * dz;

      if (dist_sq <= r2) {
        int idx = atomicAdd(collision_count, 1);
        if (idx < max_collisions) {
          collisions[2 * idx + 0] = i;
          collisions[2 * idx + 1] = j;
        }
      }
    }
  }
}

} // extern "C"

#endif // SPATIAL_OPS_CU