#!/usr/bin/env python3
"""Quick example of using the Hopfield Network toolkit."""

import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion

def basic_example():
    """Basic pattern storage and recall."""
    print("=== Basic Hopfield Network Example ===")
    
    # Create a network
    network = HopfieldNetwork(n_neurons=25, name="BasicExample")
    
    # Create some patterns (5x5 images)
    patterns = np.array([
        # Letter 'T'
        [1, 1, 1, 1, 1,
         -1, -1, 1, -1, -1,
         -1, -1, 1, -1, -1,
         -1, -1, 1, -1, -1,
         -1, -1, 1, -1, -1],
        
        # Letter 'L'
        [1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, 1, 1, 1, 1]
    ])
    
    # Train the network
    network.train_hebbian(patterns)
    
    # Add noise to a pattern
    noisy_T = patterns[0].copy()
    noise_mask = np.random.random(25) < 0.2  # 20% noise
    noisy_T[noise_mask] *= -1
    
    # Recall the pattern
    print("\nOriginal T pattern:")
    print(patterns[0].reshape(5, 5))
    print("\nNoisy T pattern:")
    print(noisy_T.reshape(5, 5))
    
    recalled = network.recall(noisy_T, max_iter=10)
    print("\nRecalled pattern:")
    print(recalled.reshape(5, 5))
    
    # Check if recall was successful
    if np.array_equal(recalled, patterns[0]):
        print("\n✓ Pattern successfully recalled!")
    else:
        print("\n✗ Pattern recall failed")

def password_example():
    """Password recovery example."""
    print("\n\n=== Password Recovery Example ===")
    
    # Create password recovery system
    recovery = PasswordRecovery(password_length=8)
    
    # Store some passwords
    passwords = ["secure12", "pass1234", "hello123"]
    recovery.train_passwords(passwords)
    
    # Try to recover a corrupted password
    corrupted = "s_cure1_"  # Missing characters
    recovered = recovery.recover_password(corrupted)
    
    print(f"Corrupted: {corrupted}")
    print(f"Recovered: {recovered}")

def pattern_completion_example():
    """Pattern completion example."""
    print("\n\n=== Pattern Completion Example ===")
    
    # Create pattern completion system
    completion = PatternCompletion(
        sequence_length=5,
        vocabulary=['A', 'B', 'C', 'D', 'E']
    )
    
    # Train on sequences
    sequences = [
        "ABCDE",
        "ACEBD",
        "BCDEA"
    ]
    completion.train_sequences(sequences)
    
    # Complete a partial sequence
    partial = "AB_D_"
    completed = completion.complete_sequence(partial)
    
    print(f"Partial:   {partial}")
    print(f"Completed: {completed}")

if __name__ == "__main__":
    basic_example()
    password_example()
    pattern_completion_example()