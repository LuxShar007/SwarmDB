import numpy as np
from src.kernel import kinetic_join_baseline

class RobotSwarm:
    def __init__(self, num_robots: int, bounds: tuple = (0.0, 1000.0)):
        """
        Initializes a swarm of robots with random (x, y, z) coordinates.
        
        Args:
            num_robots: Number of robots in the swarm.
            bounds: Tuple representing (min, max) for the coordinate bounds in all 3 dimensions.
        """
        self.num_robots = num_robots
        self.bounds = bounds
        
        # Generate N robots with (x, y, z) coordinates uniformly distributed within bounds
        self.positions = np.random.uniform(
            low=bounds[0], 
            high=bounds[1], 
            size=(num_robots, 3)
        )
        
    def update_positions(self, max_step: float = 1.0):
        """
        Simulates random movement of the swarm in 3D space.
        
        Args:
            max_step: Maximum distance a robot can travel in any dimension per tick.
        """
        steps = np.random.uniform(-max_step, max_step, size=(self.num_robots, 3))
        self.positions += steps
        # Constrain robots to stay within the simulation bounds
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

    def find_collisions(self, radius: float) -> np.ndarray:
        """
        Performs a 'Kinetic Join' to find all robots within the collision radius.
        
        Args:
            radius: The threshold distance for a collision.
            
        Returns:
            An (M, 2) numpy array of robot ID pairs (indices) that are colliding.
        """
        return kinetic_join_baseline(self.positions, radius)
