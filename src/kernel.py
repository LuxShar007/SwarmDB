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
    # Calculate the squared distances to avoid expensive square root operations.
    # We use the expansion (A - B)^2 = A^2 + B^2 - 2 A.B to compute the distance
    # matrix efficiently without creating an intermediate (N, N, 3) array.
    
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
