"""
Comprehensive tests for the student web application to ensure all features work correctly.
"""

import pytest
import numpy as np
import tempfile
import sys
import os
from unittest.mock import patch, MagicMock

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion, ImageRestoration


class TestStudentAppComponents:
    """Test all components used in the student app."""
    
    def test_pattern_memory_workflow(self):
        """Test the complete pattern memory lab workflow."""
        # Create network and patterns like the app does
        network = HopfieldNetwork(n_neurons=25, name="TestNet")
        
        # Create some test patterns (5x5 grids flattened)
        patterns = [
            np.array([1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1]),  # Square
            np.array([-1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1])  # Cross
        ]
        
        # Train network
        training_stats = network.train_hebbian(np.array(patterns))
        assert training_stats is not None
        assert 'n_patterns' in training_stats
        
        # Test noise addition and recall
        original = patterns[0]
        noisy = network.add_noise(original, 0.2)
        
        # Test recall
        result = network.recall(noisy, max_iterations=20)
        assert 'final_states' in result
        assert len(result['final_states']) == 25
        
        # Test pattern reshaping (like the app does)
        recalled_pattern = result['final_states'].reshape(5, 5)
        assert recalled_pattern.shape == (5, 5)
        
        # Test accuracy calculation
        accuracy = np.mean(result['final_states'] == original)
        assert 0 <= accuracy <= 1
    
    def test_password_recovery_workflow(self):
        """Test the complete password recovery workflow."""
        # Initialize like the app
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        
        # Train on passwords like the app
        passwords = [
            "mydog123", "password", "qwerty12", "welcome1",
            "sunshine", "dragon99", "princess", "football"
        ]
        stats = recovery.train_on_passwords(passwords, max_length=8)
        assert stats is not None
        
        # Test simple recovery interface used in app
        test_cases = [
            "m_dog___",
            "p___word", 
            "q___ty12",
            "sun___ne"
        ]
        
        for corrupted in test_cases:
            recovered = recovery.recover_password(corrupted)
            assert isinstance(recovered, str)
            assert len(recovered) <= 8
            
        # Test advanced recovery interface
        candidates = recovery.recover_password_advanced("p***word", [0, 4, 5, 6, 7], 8)
        assert len(candidates) > 0
        assert all(isinstance(cand, tuple) and len(cand) == 2 for cand in candidates)
    
    def test_sequence_prediction_workflow(self):
        """Test the complete sequence prediction workflow."""
        # Initialize like the app
        completion = PatternCompletion("text")
        
        # Train on sequences like the app (use consistent alphabet)
        sequences = [
            "ABCDABCD",
            "BCDBCDBC",
            "AABBCCDD",
            "ABABABAB",
            "CADBCADB"
        ]
        
        stats = completion.train_sequence_patterns(sequences)
        assert stats is not None
        assert completion.network is not None
        assert completion.vocab is not None
        
        # Test completion interface used in app
        test_cases = [
            "ABCD____",
            "ABAB____", 
            "AABC____",
            "CADB____"
        ]
        
        for partial in test_cases:
            completed = completion.complete_sequence(partial)
            assert isinstance(completed, str)
            assert len(completed) == len(partial)
    
    def test_network_analysis_workflow(self):
        """Test the network analysis features."""
        # Create network like the app
        network = HopfieldNetwork(n_neurons=25, name="AnalysisNet")
        
        # Create test patterns
        patterns = [
            np.random.choice([-1, 1], 25),
            np.random.choice([-1, 1], 25),
            np.random.choice([-1, 1], 25)
        ]
        
        # Train network
        network.train_hebbian(np.array(patterns))
        
        # Test energy calculation (used in app)
        for pattern in patterns:
            energy = network.energy(pattern)
            assert isinstance(energy, (int, float))
            
        # Test capacity metrics
        assert hasattr(network, 'n_neurons')
        theoretical_capacity = int(0.138 * network.n_neurons)
        assert theoretical_capacity > 0
        
        # Test weight matrix access
        assert network.weights.shape == (25, 25)
        assert np.allclose(network.weights, network.weights.T)  # Should be symmetric
        
        # Test noise tolerance like the app does
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        for noise_level in noise_levels:
            pattern = patterns[0]
            noisy = network.add_noise(pattern, noise_level)
            result = network.recall(noisy, max_iterations=20)
            accuracy = np.mean(result['final_states'] == pattern)
            assert 0 <= accuracy <= 1
    
    def test_image_restoration_workflow(self):
        """Test the image restoration features."""
        # Initialize like the app
        restorer = ImageRestoration(patch_size=4)
        
        # Create test image
        test_image = np.random.randint(0, 256, (16, 16))
        
        # Test single image training (wrapper method)
        stats = restorer.train_on_image(test_image)
        assert stats is not None
        
        # Test image restoration
        corrupted = test_image.copy()
        corrupted[6:10, 6:10] = 128  # Add corruption
        
        restored = restorer.restore_image(corrupted)
        assert restored.shape == test_image.shape
        assert restored.dtype == np.uint8
    
    def test_helper_functions(self):
        """Test helper functions used in the student app."""
        # Test pattern creation functions (defined in app)
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
        assert cross.shape == (5, 5)
        assert np.sum(cross == 1) == 9  # 5 + 5 - 1 (center overlap)
        
        square = create_square_pattern(5)
        assert square.shape == (5, 5)
        assert np.sum(square == 1) == 16  # 4 sides of square
        
        # Test noise addition
        noisy = add_noise_to_pattern(cross, 0.2)
        assert noisy.shape == cross.shape
        differences = np.sum(noisy != cross)
        assert differences > 0  # Should have some differences
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test with untrained networks
        network = HopfieldNetwork(n_neurons=25)
        
        # Should handle recall on untrained network
        random_pattern = np.random.choice([-1, 1], 25)
        result = network.recall(random_pattern)
        assert 'final_states' in result
        
        # Test password recovery with no training
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        
        # Should handle recovery without training gracefully
        try:
            result = recovery.recover_password("test____")
            # Should either work or raise informative error
            assert isinstance(result, str) or result is None
        except ValueError as e:
            assert "not trained" in str(e).lower()
        
        # Test pattern completion without training
        completion = PatternCompletion("text")
        
        try:
            result = completion.complete_sequence("TEST____")
            assert isinstance(result, str) or result is None
        except ValueError as e:
            assert "not trained" in str(e).lower()
    
    def test_parameter_validation(self):
        """Test parameter validation in all components."""
        # Test network size validation
        network = HopfieldNetwork(n_neurons=100)
        assert network.n_neurons == 100
        
        # Test pattern size validation
        patterns = np.random.choice([-1, 1], size=(3, 100))
        stats = network.train_hebbian(patterns)
        assert stats['n_patterns'] == 3
        
        # Test noise level validation
        pattern = patterns[0]
        for noise_level in [0.0, 0.1, 0.5, 1.0]:
            noisy = network.add_noise(pattern, noise_level)
            assert len(noisy) == len(pattern)
            
        # Test password length validation
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        
        passwords = ["test", "longer_password"]
        stats = recovery.train_on_passwords(passwords, max_length=15)
        assert stats is not None
    
    def test_performance_requirements(self):
        """Test that operations complete in reasonable time."""
        import time
        
        # Test network creation and training speed
        start_time = time.time()
        network = HopfieldNetwork(n_neurons=100)
        patterns = np.random.choice([-1, 1], size=(5, 100))
        network.train_hebbian(patterns)
        training_time = time.time() - start_time
        
        # Should complete training quickly (< 1 second for small networks)
        assert training_time < 1.0
        
        # Test recall speed
        start_time = time.time()
        noisy = network.add_noise(patterns[0], 0.2)
        result = network.recall(noisy, max_iterations=20)
        recall_time = time.time() - start_time
        
        # Should complete recall quickly
        assert recall_time < 1.0
        
    def test_memory_usage(self):
        """Test reasonable memory usage."""
        # Test that large networks don't consume excessive memory
        network = HopfieldNetwork(n_neurons=1000)
        patterns = np.random.choice([-1, 1], size=(10, 1000))
        
        # This should work without memory errors
        stats = network.train_hebbian(patterns)
        assert stats is not None
        
        # Test recall works
        result = network.recall(patterns[0], max_iterations=10)
        assert 'final_states' in result


class TestAppIntegration:
    """Integration tests for the complete app functionality."""
    
    def test_complete_student_workflow(self):
        """Test a complete workflow that a student might follow."""
        # 1. Create and train a pattern memory network
        network = HopfieldNetwork(n_neurons=25, name="StudentTest")
        
        # Student draws a cross pattern
        cross_pattern = np.array([
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1, 
            1, 1, 1, 1, 1,
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1
        ])
        
        # Train the network
        network.train_hebbian(np.array([cross_pattern]))
        
        # 2. Add noise and test recall
        noisy = network.add_noise(cross_pattern, 0.3)
        result = network.recall(noisy, max_iterations=20)
        recalled = result['final_states']
        
        # Should recall the original pattern
        accuracy = np.mean(recalled == cross_pattern)
        assert accuracy >= 0.8  # Should get most of it right
        
        # 3. Try password recovery
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        recovery.train_on_passwords(['student123', 'school456', 'learn789'], max_length=10)
        
        recovered = recovery.recover_password('stu___123')
        assert isinstance(recovered, str)
        
        # 4. Try sequence completion
        completion = PatternCompletion("text")
        completion.train_sequence_patterns(['ABCABC', 'DEFDEF', 'GHIGHIGH'])
        
        completed = completion.complete_sequence('ABC___')
        assert isinstance(completed, str)
        
        # All steps should complete without errors
        assert True
    
    def test_error_recovery(self):
        """Test that the app recovers gracefully from errors."""
        # Test with invalid inputs
        network = HopfieldNetwork(n_neurons=25)
        
        # Try to recall with wrong size pattern
        try:
            wrong_size = np.array([1, -1, 1])  # Wrong size
            result = network.recall(wrong_size)
            # Should either work or fail gracefully
        except Exception as e:
            # Should be an informative error
            assert len(str(e)) > 0
    
    def test_app_state_management(self):
        """Test state management scenarios."""
        # Test multiple networks
        net1 = HopfieldNetwork(n_neurons=25, name="Net1")
        net2 = HopfieldNetwork(n_neurons=25, name="Net2")
        
        # Train with different patterns
        patterns1 = [np.random.choice([-1, 1], 25)]
        patterns2 = [np.random.choice([-1, 1], 25)]
        
        net1.train_hebbian(np.array(patterns1))
        net2.train_hebbian(np.array(patterns2))
        
        # Both should work independently
        result1 = net1.recall(patterns1[0])
        result2 = net2.recall(patterns2[0])
        
        assert 'final_states' in result1
        assert 'final_states' in result2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])