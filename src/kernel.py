import numpy as np

def kinetic_join_baseline(positions: np.ndarray, radius: float) -> np.ndarray:
    """
    Computes all pairs of robots within the specified collision radius.
    
    This is an O(N^2) baseline implementation using pure NumPy vectorization.
    It simulates the computational bottleneck that will be offloaded to CUDA.
    
    Args:
        positions: An (N, 3) numpy array of robot coordinates.
        radius: The collision threshold radius.
        
    Returns:
        An (M, 2) numpy array of indices representing colliding pairs.
    """
    sq_norms = np.sum(positions**2, axis=1)
    
    # Compute the N x N pairwise squared distance matrix
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(positions, positions.T)
    
    # Address floating point inaccuracies that might result in small negative numbers
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # We are only interested in unique pairs (i, j) where i < j
    # Set the lower triangle and the diagonal to infinity to ignore them
    dist_sq[np.tril_indices(positions.shape[0])] = np.inf
    
    radius_sq = radius**2
    
    # Find indices where squared distance is less than or equal to squared radius
    i, j = np.where(dist_sq <= radius_sq)
    
    return np.column_stack((i, j))


def kinetic_join_optimized(positions: np.ndarray, radius: float) -> np.ndarray:
    """
    Computes all pairs of robots within the specified collision radius.
    
    This is an optimized implementation using SciPy's cKDTree (spatial indexing).
    Complexity: O(N log N) time, O(N) memory.
    
    Args:
        positions: An (N, 3) numpy array of robot coordinates.
        radius: The collision threshold radius.
        
    Returns:
        An (M, 2) numpy array of indices representing colliding pairs.
    """
    from scipy.spatial import cKDTree
    
    if positions.shape[0] == 0:
        return np.empty((0, 2), dtype=int)
        
    tree = cKDTree(positions)
    pairs = tree.query_pairs(radius)
    
    return np.array(list(pairs)) if len(pairs) > 0 else np.empty((0, 2), dtype=int)


def kinetic_join_cuda(positions: np.ndarray, radius: float) -> np.ndarray:
    """
    Computes all pairs of robots within the specified collision radius using CUDA.
    Loads and compiles the CUDA kernel in src/kernels/spatial_ops.cu.
    
    Args:
        positions: An (N, 3) numpy array of robot coordinates.
        radius: The collision threshold radius.
        
    Returns:
        An (M, 2) numpy array of indices representing colliding pairs.
    """
    import os
    
    cuda_path = os.path.join(os.path.dirname(__file__), "kernels", "spatial_ops.cu")
    if not os.path.exists(cuda_path):
        raise FileNotFoundError(f"CUDA kernel file not found at {cuda_path}")
        
    with open(cuda_path, "r") as f:
        cuda_src = f.read()

    num_robots = positions.shape[0]
    if num_robots == 0:
        return np.empty((0, 2), dtype=int)

    # 1. Attempt using CuPy
    try:
        import cupy as cp  # type: ignore
        
        module = cp.RawModule(code=cuda_src)
        kernel = module.get_function("kinetic_join_shared_mem_kernel")
        
        pos_gpu = cp.array(positions.astype(np.float32))
        collision_count_gpu = cp.zeros(1, dtype=np.int32)
        
        # Upper bound limit for collisions
        max_collisions = min(1000000, max(1000, num_robots * 100))
        collisions_gpu = cp.zeros(max_collisions * 2, dtype=np.int32)
        
        block_size = 256
        grid_size = (num_robots + block_size - 1) // block_size
        
        kernel(
            (grid_size,), (block_size,),
            (pos_gpu, np.int32(num_robots), np.float32(radius), collision_count_gpu, collisions_gpu, np.int32(max_collisions))
        )
        
        count = int(collision_count_gpu[0])
        count = min(count, max_collisions)
        
        if count == 0:
            return np.empty((0, 2), dtype=int)
            
        res = collisions_gpu[:count * 2].get()
        return res.reshape((count, 2))
        
    except (ImportError, Exception):
        pass

    # 2. Attempt using PyCUDA
    try:
        import pycuda.autoinit  # type: ignore
        import pycuda.driver as cuda  # type: ignore
        from pycuda.compiler import SourceModule  # type: ignore
        
        mod = SourceModule(cuda_src)
        kernel = mod.get_function("kinetic_join_shared_mem_kernel")
        
        pos_float32 = positions.astype(np.float32)
        pos_gpu = cuda.mem_alloc(pos_float32.nbytes)
        cuda.memcpy_htod(pos_gpu, pos_float32)
        
        collision_count = np.zeros(1, dtype=np.int32)
        collision_count_gpu = cuda.mem_alloc(collision_count.nbytes)
        cuda.memcpy_htod(collision_count_gpu, collision_count)
        
        max_collisions = min(1000000, max(1000, num_robots * 100))
        collisions = np.zeros(max_collisions * 2, dtype=np.int32)
        collisions_gpu = cuda.mem_alloc(collisions.nbytes)
        cuda.memcpy_htod(collisions_gpu, collisions)
        
        block_size = 256
        grid_size = (num_robots + block_size - 1) // block_size
        
        kernel(
            pos_gpu, np.int32(num_robots), np.float32(radius), 
            collision_count_gpu, collisions_gpu, np.int32(max_collisions),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )
        
        cuda.memcpy_dtoh(collision_count, collision_count_gpu)
        count = int(collision_count[0])
        count = min(count, max_collisions)
        
        if count == 0:
            return np.empty((0, 2), dtype=int)
            
        cuda.memcpy_dtoh(collisions, collisions_gpu)
        res = collisions[:count * 2]
        return res.reshape((count, 2))
        
    except (ImportError, Exception) as e:
        raise ImportError(f"CUDA execution failed. Neither CuPy nor PyCUDA could be initialized, or: {str(e)}")


# Lazy-loaded CuPy Elementwise Kernel (from user's code)
cupy_elementwise_kernel = None

def get_cupy_elementwise_kernel():
    global cupy_elementwise_kernel
    if cupy_elementwise_kernel is None:
        import cupy as cp  # type: ignore
        cupy_elementwise_kernel = cp.ElementwiseKernel(
            in_params='raw float32 positions, raw float32 sq_norms, float32 radius_sq, int32 N',
            out_params='raw int32 collision_matrix',
            operation='''
                // Get the global 1D index of the current GPU thread
                int idx = i; 
                
                // Map the 1D thread index to an (row, col) coordinate in our theoretical matrix
                int row = idx / N;
                int col = idx % N;
                
                // We only care about unique pairs where row < col (the upper triangle)
                if (row < col) {
                    // Grab coordinates for Robot A (row)
                    float ax = positions[row * 3 + 0];
                    float ay = positions[row * 3 + 1];
                    float az = positions[row * 3 + 2];
                    
                    // Grab coordinates for Robot B (col)
                    float bx = positions[col * 3 + 0];
                    float by = positions[col * 3 + 1];
                    float bz = positions[col * 3 + 2];
                    
                    // Compute the distance using standard 3D Euclidean space math
                    float dx = ax - bx;
                    float dy = ay - by;
                    float dz = az - bz;
                    float actual_dist_sq = (dx * dx) + (dy * dy) + (dz * dz);
                    
                    // If they are colliding, flag it with a 1 in our output matrix
                    if (actual_dist_sq <= radius_sq) {
                        collision_matrix[idx] = 1;
                    }
                }
            ''',
            name='cuda_kinetic_join'
        )
    return cupy_elementwise_kernel


def run_gpu_elementwise(positions: np.ndarray, radius: float) -> np.ndarray:
    """
    User's CuPy Elementwise implementation.
    Launches the Elementwise CUDA kernel dynamically if CuPy is installed.
    """
    import cupy as cp  # type: ignore
    N = positions.shape[0]
    radius_sq = float(radius ** 2)
    
    # 1. Transfer coordinate matrices directly to GPU VRAM memory space
    positions_gpu = cp.asarray(positions, dtype=cp.float32)
    sq_norms_gpu = cp.sum(positions_gpu**2, axis=1)
    
    # 2. Allocate an output flag matrix on the GPU
    collision_matrix_gpu = cp.zeros((N * N,), dtype=cp.int32)
    
    # 3. Get and fire the Elementwise kernel
    kernel = get_cupy_elementwise_kernel()
    kernel(positions_gpu, sq_norms_gpu, radius_sq, N, collision_matrix_gpu, size=N*N)
    
    # Find the matching indices
    flat_indices = cp.where(collision_matrix_gpu == 1)[0]
    i = flat_indices // N
    j = flat_indices % N
    
    return cp.column_stack((i, j))