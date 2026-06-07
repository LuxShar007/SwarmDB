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
#endif

#define BLOCK_SIZE 256

extern "C" {

/**
 * Naive Kinetic Join CUDA Kernel
 * Assigns one thread per robot i. Thread i iterates over all robots j > i to
 * compute squared distances. Collision indices are written atomically into the
 * global collisions array.
 */
__global__ void kinetic_join_naive_kernel(
    const float *__restrict__ positions, // N * 3 flat array [x0, y0, z0, x1,
                                         // y1, z1, ...]
    int num_robots, float radius, int *__restrict__ collision_count,
    int *__restrict__ collisions, // flat output array of size max_collisions *
                                  // 2 [i0, j0, i1, j1, ...]
    int max_collisions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_robots)
    return;

  float r2 = radius * radius;
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

/**
 * Shared Memory Optimized Kinetic Join CUDA Kernel
 * Divides robots into blocks of size BLOCK_SIZE and loads blocks of positions
 * into shared memory to minimize global memory bandwidth saturation during
 * collision calculations.
 */
__global__ void kinetic_join_shared_mem_kernel(
    const float *__restrict__ positions, // N * 3 flat array
    int num_robots, float radius, int *__restrict__ collision_count,
    int *__restrict__ collisions, // flat output array
    int max_collisions) {
  __shared__ float shared_pos[BLOCK_SIZE * 3];

  int tx = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tx;

  float r2 = radius * radius;
  float xi = 0.0f, yi = 0.0f, zi = 0.0f;

  if (i < num_robots) {
    xi = positions[3 * i + 0];
    yi = positions[3 * i + 1];
    zi = positions[3 * i + 2];
  }

  // Loop over all blocks of robots
  int total_blocks = (num_robots + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int b = 0; b < total_blocks; ++b) {
    // Load positions of block b into shared memory
    int load_idx = b * BLOCK_SIZE + tx;
    if (load_idx < num_robots) {
      shared_pos[3 * tx + 0] = positions[3 * load_idx + 0];
      shared_pos[3 * tx + 1] = positions[3 * load_idx + 1];
      shared_pos[3 * tx + 2] = positions[3 * load_idx + 2];
    } else {
      shared_pos[3 * tx + 0] = 0.0f;
      shared_pos[3 * tx + 1] = 0.0f;
      shared_pos[3 * tx + 2] = 0.0f;
    }
    __syncthreads();

    if (i < num_robots) {
      // Compute distances against all robots in this block
      int limit = min(BLOCK_SIZE, num_robots - b * BLOCK_SIZE);
      for (int k = 0; k < limit; ++k) {
        int j = b * BLOCK_SIZE + k;
        if (i < j) { // Only calculate unique pairs (i < j)
          float dx = xi - shared_pos[3 * k + 0];
          float dy = yi - shared_pos[3 * k + 1];
          float dz = zi - shared_pos[3 * k + 2];
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
    __syncthreads();
  }
}

} // extern "C"

#endif // SPATIAL_OPS_CU
