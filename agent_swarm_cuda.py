import cupy as cp
import numpy as np
import time

# 1. Define a raw CUDA C++ Kernel inside a Python string block.
# This runs directly inside individual streaming multiprocessors on your GPU!
kinetic_join_kernel = cp.ElementwiseKernel(
    in_params='raw float32 positions, float32 radius_sq, int32 N',
    out_params='raw int32 collision_matrix',
    operation='''
        // 'i' is an automatically provided variable representing the global 1D thread ID.
        int thread_idx = i; 
        
        // Map our flat 1D thread layout to a theoretical (row, col) matrix index
        int row = thread_idx / N;
        int col = thread_idx % N;
        
        // Symmetric Optimization: Only compute the Upper Triangle (row < col) 
        // to completely cut computation work in half!
        if (row < col) {
            // Index directly into our raw 1D position array pointers (X, Y, Z coordinates)
            float ax = positions[row * 3 + 0];
            float ay = positions[row * 3 + 1];
            float az = positions[row * 3 + 2];
            
            float bx = positions[col * 3 + 0];
            float by = positions[col * 3 + 1];
            float bz = positions[col * 3 + 2];
            
            // Core 3D Euclidean distance calculations running completely in hardware registers
            float dx = ax - bx;
            float dy = ay - by;
            float dz = az - bz;
            float actual_dist_sq = (dx * dx) + (dy * dy) + (dz * dz);
            
            // If the spatial boundaries intersect, mark it with a 1 flag
            if (actual_dist_sq <= radius_sq) {
                collision_matrix[thread_idx] = 1;
            }
        }
    ''',
    name='cuda_kinetic_join'
)

def run_accelerated_simulation(N, radius):
    radius_sq = float(radius ** 2)
    
    # Generate random robot positions
    print(f"[*] Simulating {N:,} robot coordinate vectors...")
    mock_positions = np.random.uniform(-1000, 1000, (N, 3)).astype(np.float32)
    
    # Push the coordinate block straight to VRAM
    positions_gpu = cp.asarray(mock_positions)
    
    # Allocate a flat tracking array on the GPU
    collision_matrix_gpu = cp.zeros((N * N,), dtype=cp.int32)
    
    print(f"🚀 Launching custom compiled CUDA C++ kernel across {N*N:,} threads...")
    start_time = time.time()
    
    # Invoke the kernel hardware grid loop
    kinetic_join_kernel(positions_gpu, radius_sq, N, collision_matrix_gpu, size=N*N)
    
    # Force the host CPU to wait for the async GPU threads to finish aligning
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    # Extract structural coordinate indices from the flag matrix
    flat_indices = cp.where(collision_matrix_gpu == 1)[0]
    i_indices = flat_indices // N
    j_indices = flat_indices % N
    colliding_pairs = cp.column_stack((i_indices, j_indices))
    
    print("\n=== CUDA Core Compute Kernel Diagnostics ===")
    print(f"| Entities Processed:   {N:,}")
    print(f"| Total Spawning Threads:{N*N:,}")
    print(f"| Latency Overhead:     {end_time - start_time:.6f} seconds")
    print(f"| Extracted Collisions: {len(colliding_pairs)}")
    print("============================================\n")

if __name__ == "__main__":
    print("============================================\n")
    print("[*] WARM-UP RUN (Including JIT Compilation)...")
    run_accelerated_simulation(N=10000, radius=25.0)
    
    print("\n" + "="*44 + "\n")
    print("[*] BENCHMARK RUN 2 (Pure Hardware Speed)...")
    run_accelerated_simulation(N=10000, radius=25.0)
    
    print("\n" + "="*44 + "\n")
    print("[*] STRESS TEST RUN 3 (Pushing Entity Scaling)...")
    run_accelerated_simulation(N=15000, radius=25.0)