import numpy as np
from typing import List, Tuple, Optional
import time


class HopfieldNetwork:
    """Fast Hopfield Network implementation using NumPy for vectorized operations."""
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.states = np.random.choice([-1, 1], size=n_neurons)
        self.energy_history = []
        
    def train(self, patterns: np.ndarray):
        """Train the network using Hebbian learning rule.
        
        Args:
            patterns: Array of shape (n_patterns, n_neurons) with values in {-1, 1}
        """
        n_patterns = patterns.shape[0]
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
            
        self.weights /= n_patterns
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
    def set_weights(self, connections: List[Tuple[int, int, float]]):
        """Manually set connection weights (for compatibility with Java version).
        
        Args:
            connections: List of (neuron_i, neuron_j, weight) tuples
        """
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        for i, j, weight in connections:
            self.weights[i, j] = weight
            self.weights[j, i] = weight  # Symmetric connections
            
    def energy(self) -> float:
        """Calculate the network's energy."""
        return -0.5 * np.dot(self.states, np.dot(self.weights, self.states))
    
    def update_async(self, max_iterations: int = 100) -> int:
        """Update neurons asynchronously until convergence.
        
        Returns:
            Number of iterations until convergence
        """
        self.energy_history = [self.energy()]
        
        for iteration in range(max_iterations):
            changed = False
            indices = np.random.permutation(self.n_neurons)
            
            for i in indices:
                local_field = np.dot(self.weights[i], self.states)
                new_state = 1 if local_field > 0 else -1
                
                if new_state != self.states[i]:
                    self.states[i] = new_state
                    changed = True
                    
            self.energy_history.append(self.energy())
            
            if not changed:
                return iteration + 1
                
        return max_iterations
    
    def update_sync(self) -> bool:
        """Update all neurons synchronously.
        
        Returns:
            True if any state changed
        """
        local_fields = np.dot(self.weights, self.states)
        new_states = np.where(local_fields > 0, 1, -1)
        changed = not np.array_equal(new_states, self.states)
        self.states = new_states
        return changed
    
    def recall(self, partial_pattern: np.ndarray, async_update: bool = True, 
               max_iterations: int = 100) -> np.ndarray:
        """Recall a complete pattern from a partial/noisy input.
        
        Args:
            partial_pattern: Input pattern
            async_update: Use asynchronous (True) or synchronous (False) updates
            max_iterations: Maximum update iterations
            
        Returns:
            Recalled pattern
        """
        self.states = partial_pattern.copy()
        
        if async_update:
            self.update_async(max_iterations)
        else:
            for _ in range(max_iterations):
                if not self.update_sync():
                    break
                    
        return self.states.copy()
    
    def add_noise(self, pattern: np.ndarray, noise_level: float) -> np.ndarray:
        """Add noise to a pattern by flipping bits.
        
        Args:
            pattern: Original pattern
            noise_level: Fraction of bits to flip (0-1)
            
        Returns:
            Noisy pattern
        """
        noisy = pattern.copy()
        n_flip = int(noise_level * len(pattern))
        flip_indices = np.random.choice(len(pattern), n_flip, replace=False)
        noisy[flip_indices] *= -1
        return noisy


def create_example_network() -> HopfieldNetwork:
    """Create the 4-neuron network from the Java example."""
    network = HopfieldNetwork(4)
    # Neurons: A=0, B=1, C=2, D=3
    connections = [
        (0, 1, -1),  # A-B: -1
        (1, 2, 1),   # B-C: 1
        (1, 3, -1),  # B-D: -1
        (0, 3, 1),   # A-D: 1
    ]
    network.set_weights(connections)
    return network


if __name__ == "__main__":
    # Test with the Java example
    print("Creating 4-neuron network from Java example...")
    network = create_example_network()
    
    print(f"Initial states: {network.states}")
    print(f"Initial energy: {network.energy():.2f}")
    
    iterations = network.update_async()
    print(f"\nConverged after {iterations} iterations")
    print(f"Final states: {network.states}")
    print(f"Final energy: {network.energy():.2f}")
    
    # Test pattern storage and recall
    print("\n" + "="*50)
    print("Testing pattern storage and recall...")
    
    # Create a larger network
    n = 25
    network = HopfieldNetwork(n)
    
    # Create some patterns to store
    patterns = np.array([
        [1, 1, 1, -1, -1] * 5,  # Stripe pattern
        [1, -1, 1, -1, 1] * 5,  # Alternating pattern
    ])
    
    network.train(patterns)
    
    # Test recall with noise
    original = patterns[0]
    noisy = network.add_noise(original, 0.2)
    recalled = network.recall(noisy)
    
    print(f"Original pattern: {original[:10]}...")
    print(f"Noisy pattern:    {noisy[:10]}...")
    print(f"Recalled pattern: {recalled[:10]}...")
    print(f"Recall accuracy: {np.mean(recalled == original):.2%}")