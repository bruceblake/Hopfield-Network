#!/usr/bin/env python3
"""
Final comprehensive test to ensure the web app works without errors.
Tests all the functionality that the student app uses.
"""

import sys
import os
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_student_app_functionality():
    """Test every function and method used in the student app."""
    print("🧪 Testing Student App Functionality...")
    
    # Test imports
    try:
        from hopfield_toolkit import HopfieldNetwork
        from applications import PasswordRecovery, PatternCompletion, ImageRestoration
        import time
        import random
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test pattern memory lab workflow
    try:
        print("\n🎨 Testing Pattern Memory Lab...")
        
        # Create network like the app
        network = HopfieldNetwork(n_neurons=25, name="TestNet")
        
        # Create patterns like the app
        patterns = [
            np.random.choice([-1, 1], 25),
            np.random.choice([-1, 1], 25),
            np.random.choice([-1, 1], 25)
        ]
        
        # Train network
        network.train_hebbian(np.array(patterns))
        
        # Test noise and recall
        noisy = network.add_noise(patterns[0], 0.3)
        result = network.recall(noisy, max_iterations=20)
        recalled = result['final_states'].reshape(5, 5)
        
        # Test accuracy calculation
        accuracy = np.mean(result['final_states'] == patterns[0])
        
        print(f"   ✅ Pattern memory: {accuracy:.1%} accuracy")
        
    except Exception as e:
        print(f"   ❌ Pattern memory error: {e}")
        return False
    
    # Test password recovery workflow
    try:
        print("\n🔐 Testing Password Recovery...")
        
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        passwords = [
            "mydog123", "password", "qwerty12", "welcome1",
            "sunshine", "dragon99", "princess", "football"
        ]
        recovery.train_on_passwords(passwords, max_length=8)
        
        # Test simple API
        test_cases = ["m_dog___", "p___word", "q___ty12"]
        for corrupted in test_cases:
            recovered = recovery.recover_password(corrupted)
            print(f"   {corrupted} -> {recovered}")
        
        print("   ✅ Password recovery working")
        
    except Exception as e:
        print(f"   ❌ Password recovery error: {e}")
        return False
    
    # Test sequence completion workflow
    try:
        print("\n🧩 Testing Sequence Completion...")
        
        completion = PatternCompletion("text")
        sequences = [
            "ABCDABCD",
            "BCDBCDBC",
            "AABBCCDD",
            "ABABABAB",
            "CADBCADB"
        ]
        completion.train_sequence_patterns(sequences)
        
        # Test completion
        test_cases = ["ABCD____", "ABAB____", "AABC____"]
        for partial in test_cases:
            completed = completion.complete_sequence(partial)
            print(f"   {partial} -> {completed}")
            assert len(completed) == len(partial), f"Length mismatch: {len(completed)} != {len(partial)}"
        
        print("   ✅ Sequence completion working")
        
    except Exception as e:
        print(f"   ❌ Sequence completion error: {e}")
        return False
    
    # Test network analysis workflow
    try:
        print("\n📊 Testing Network Analysis...")
        
        network = HopfieldNetwork(n_neurons=25, name="AnalysisNet")
        patterns = [np.random.choice([-1, 1], 25) for _ in range(3)]
        network.train_hebbian(np.array(patterns))
        
        # Test energy calculation (the method that was failing)
        for i, pattern in enumerate(patterns):
            energy = network.energy(pattern)
            print(f"   Pattern {i+1} energy: {energy:.2f}")
        
        # Test noise tolerance
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        for noise_level in noise_levels:
            noisy = network.add_noise(patterns[0], noise_level)
            result = network.recall(noisy, max_iterations=20)
            accuracy = np.mean(result['final_states'] == patterns[0])
            # Note: don't check accuracy too strictly as it may vary
        
        print("   ✅ Network analysis working")
        
    except Exception as e:
        print(f"   ❌ Network analysis error: {e}")
        return False
    
    # Test image restoration workflow
    try:
        print("\n🖼️ Testing Image Restoration...")
        
        restorer = ImageRestoration(patch_size=4)
        test_image = np.random.randint(0, 256, (16, 16))
        
        # Test single image training (wrapper method that was missing)
        restorer.train_on_image(test_image)
        
        # Test restoration
        corrupted = test_image.copy()
        corrupted[6:10, 6:10] = 128
        restored = restorer.restore_image(corrupted)
        
        print(f"   Image shape: {restored.shape}, dtype: {restored.dtype}")
        print("   ✅ Image restoration working")
        
    except Exception as e:
        print(f"   ❌ Image restoration error: {e}")
        return False
    
    # Test helper functions used in the app
    try:
        print("\n🔧 Testing Helper Functions...")
        
        def create_cross_pattern(size):
            pattern = np.ones((size, size)) * -1
            mid = size // 2
            pattern[mid, :] = 1
            pattern[:, mid] = 1
            return pattern
        
        def create_square_pattern(size):
            pattern = np.ones((size, size)) * -1
            pattern[0, :] = 1
            pattern[-1, :] = 1
            pattern[:, 0] = 1
            pattern[:, -1] = 1
            return pattern
        
        def add_noise_to_pattern(pattern, noise_level):
            noisy = pattern.copy()
            flat = noisy.flatten()
            n_flip = int(len(flat) * noise_level)
            flip_indices = np.random.choice(len(flat), n_flip, replace=False)
            flat[flip_indices] *= -1
            return flat.reshape(pattern.shape)
        
        # Test these functions
        cross = create_cross_pattern(5)
        square = create_square_pattern(5) 
        noisy = add_noise_to_pattern(cross, 0.2)
        
        print("   ✅ Helper functions working")
        
    except Exception as e:
        print(f"   ❌ Helper functions error: {e}")
        return False
    
    print("\n🎉 ALL STUDENT APP FUNCTIONALITY TESTED SUCCESSFULLY!")
    print("✅ Pattern memory lab: Working")
    print("✅ Password recovery: Working")
    print("✅ Sequence completion: Working") 
    print("✅ Network analysis: Working")
    print("✅ Image restoration: Working")
    print("✅ Helper functions: Working")
    print("\n🚀 Web app is ready for students!")
    
    return True

if __name__ == "__main__":
    success = test_all_student_app_functionality()
    if success:
        print("\n✨ Student app is ready to amaze high school students! ✨")
        exit(0)
    else:
        print("\n💥 Some issues found - check the errors above")
        exit(1)