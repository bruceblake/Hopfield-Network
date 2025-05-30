#!/usr/bin/env python3
"""
Example usage of the Hopfield Network implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork


def visualize_patterns(patterns, title="Patterns"):
    """Visualize multiple patterns in a grid."""
    n = len(patterns)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    
    for i, (ax, pattern) in enumerate(zip(axes, patterns)):
        size = int(np.sqrt(len(pattern)))
        ax.imshow(pattern.reshape(size, size), cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"Pattern {i+1}")
        ax.axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def example_1_basic():
    """Basic 4-neuron network example (Java equivalent)."""
    print("=== Example 1: Basic 4-Neuron Network ===")
    
    network = HopfieldNetwork(4)
    connections = [
        (0, 1, -1),  # A-B: inhibitory
        (1, 2, 1),   # B-C: excitatory
        (1, 3, -1),  # B-D: inhibitory
        (0, 3, 1),   # A-D: excitatory
    ]
    network.set_weights(connections)
    
    print("Initial state:", network.states)
    print("Initial energy:", network.energy())
    
    iterations = network.update_async()
    
    print(f"Converged after {iterations} iterations")
    print("Final state:", network.states)
    print("Final energy:", network.energy())
    print()


def example_2_pattern_storage():
    """Store and recall letter patterns."""
    print("=== Example 2: Letter Pattern Storage ===")
    
    # Define 5x5 letter patterns
    T = np.array([
        [1, 1, 1, 1, 1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
    ]).flatten()
    
    L = np.array([
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1],
    ]).flatten()
    
    X = np.array([
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
    ]).flatten()
    
    # Create and train network
    network = HopfieldNetwork(25)
    patterns = np.array([T, L, X])
    network.train(patterns)
    
    # Test recall with noisy T
    noise_level = 0.3
    noisy_T = network.add_noise(T, noise_level)
    recalled_T = network.recall(noisy_T)
    
    print(f"Added {noise_level:.0%} noise to pattern T")
    print(f"Recall accuracy: {np.mean(recalled_T == T):.1%}")
    print(f"Perfect recall: {'Yes' if np.array_equal(recalled_T, T) else 'No'}")
    
    # Visualize
    visualize_patterns([T, noisy_T, recalled_T], 
                      "Original T → Noisy T → Recalled T")
    print()


def example_3_capacity_test():
    """Test network capacity limits."""
    print("=== Example 3: Network Capacity Test ===")
    
    network_size = 100
    network = HopfieldNetwork(network_size)
    
    # Theoretical capacity is ~0.138N for random patterns
    theoretical_capacity = int(0.138 * network_size)
    
    print(f"Network size: {network_size} neurons")
    print(f"Theoretical capacity: ~{theoretical_capacity} patterns")
    print("\nTesting pattern storage:")
    
    # Test with increasing number of patterns
    for n_patterns in [5, 10, 15, 20]:
        # Generate random patterns
        patterns = np.random.choice([-1, 1], size=(n_patterns, network_size))
        network.train(patterns)
        
        # Test recall
        successes = 0
        for pattern in patterns:
            noisy = network.add_noise(pattern, 0.1)
            recalled = network.recall(noisy, max_iterations=50)
            if np.array_equal(recalled, pattern):
                successes += 1
        
        success_rate = successes / n_patterns
        print(f"  {n_patterns} patterns: {success_rate:.0%} perfect recall")
    print()


def example_4_energy_landscape():
    """Visualize energy landscape for small network."""
    print("=== Example 4: Energy Landscape Visualization ===")
    
    # Create 2-neuron network for visualization
    network = HopfieldNetwork(2)
    network.set_weights([(0, 1, 1)])  # Single excitatory connection
    
    # Calculate energy for all possible states
    states = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    energies = []
    
    for state in states:
        network.states = np.array(state)
        energies.append(network.energy())
    
    # Display results
    print("State configurations and their energies:")
    for state, energy in zip(states, energies):
        state_str = ''.join(['+' if s > 0 else '-' for s in state])
        print(f"  State {state_str}: Energy = {energy:.1f}")
    
    # Find stable states
    stable_states = [state for state, e in zip(states, energies) 
                    if e == min(energies)]
    print(f"\nStable states (minimum energy): {stable_states}")
    print()


def example_5_real_world_application():
    """Digit recognition example."""
    print("=== Example 5: Digit Recognition ===")
    
    # Define simple 3x3 digit patterns
    digits = {
        '0': np.array([
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ]),
        '1': np.array([
            [-1, 1, -1],
            [-1, 1, -1],
            [-1, 1, -1],
        ]),
        '7': np.array([
            [1, 1, 1],
            [-1, -1, 1],
            [-1, -1, 1],
        ]),
    }
    
    # Flatten patterns
    patterns = np.array([d.flatten() for d in digits.values()])
    
    # Create and train network
    network = HopfieldNetwork(9)
    network.train(patterns)
    
    # Create corrupted digit '0'
    corrupted = digits['0'].copy()
    corrupted[1, 1] = 1  # Fill in the center
    corrupted[0, 1] = -1  # Remove top middle
    corrupted_flat = corrupted.flatten()
    
    # Recall
    recalled = network.recall(corrupted_flat)
    
    # Check which digit was recalled
    for name, digit in digits.items():
        if np.array_equal(recalled, digit.flatten()):
            print(f"Corrupted pattern recognized as: {name}")
            break
    
    # Visualize
    patterns_to_show = [
        digits['0'].flatten(),
        corrupted_flat,
        recalled
    ]
    visualize_patterns(patterns_to_show, 
                      "Original '0' → Corrupted → Recalled")


if __name__ == "__main__":
    examples = [
        example_1_basic,
        example_2_pattern_storage,
        example_3_capacity_test,
        example_4_energy_landscape,
        example_5_real_world_application,
    ]
    
    for example in examples:
        example()
        input("Press Enter to continue to next example...")
        print()