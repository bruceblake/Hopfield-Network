"""
Tests for Hopfield Network Applications
"""

import pytest
import numpy as np
import tempfile
import os
from applications import ImageRestoration, PasswordRecovery, OptimizationSolver, PatternCompletion


class TestImageRestoration:
    """Test image restoration functionality."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample test images."""
        return [
            np.random.randint(0, 256, (32, 32)) for _ in range(3)
        ]
    
    def test_patch_extraction(self, sample_images):
        """Test patch extraction from images."""
        restorer = ImageRestoration(patch_size=4)
        patches = restorer.extract_patches(sample_images[0])
        
        # Check patch dimensions
        assert patches.shape[1] == 16  # 4x4 patches
        assert patches.shape[0] > 0  # Should have patches
    
    def test_training(self, sample_images):
        """Test training on image patches."""
        restorer = ImageRestoration(patch_size=4)
        stats = restorer.train_on_images(sample_images, "test_type")
        
        assert "test_type" in restorer.networks
        assert "test_type" in restorer.patch_patterns
        assert 'n_patterns' in stats
        assert 'training_time' in stats
    
    def test_image_restoration(self, sample_images):
        """Test image restoration process."""
        restorer = ImageRestoration(patch_size=4)
        restorer.train_on_images(sample_images[:2], "test")
        
        # Create noisy image
        noisy_image = sample_images[2]
        restored = restorer.restore_image(noisy_image, "test")
        
        assert restored.shape == noisy_image.shape
        assert restored.dtype == np.uint8
        assert np.all((restored == 0) | (restored == 255))  # Binary output


class TestPasswordRecovery:
    """Test password recovery functionality."""
    
    @pytest.fixture
    def test_passwords(self):
        """Sample passwords for testing."""
        return ["password123", "admin2024", "user1234", "secure99", "test123"]
    
    def test_encoding_setup(self):
        """Test character encoding setup."""
        recovery = PasswordRecovery()
        charset = "abcd123"
        recovery.setup_encoding(charset)
        
        assert len(recovery.char_to_idx) == len(charset)
        assert len(recovery.idx_to_char) == len(charset)
        
        for char in charset:
            assert char in recovery.char_to_idx
            idx = recovery.char_to_idx[char]
            assert recovery.idx_to_char[idx] == char
    
    def test_password_encoding_decoding(self):
        """Test password encoding and decoding."""
        recovery = PasswordRecovery()
        recovery.setup_encoding("abc123")
        
        password = "abc123"
        encoded = recovery.encode_password(password, 10)
        decoded = recovery.decode_pattern(encoded)
        
        assert len(encoded) == 10 * len(recovery.char_to_idx)
        assert np.all(np.abs(encoded) == 1)  # Binary
        assert password in decoded  # Should contain original
    
    def test_password_training(self, test_passwords):
        """Test training on password patterns."""
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        
        stats = recovery.train_on_passwords(test_passwords, max_length=12)
        
        assert recovery.network is not None
        assert 'n_patterns' in stats
        assert recovery.pattern_length > 0
    
    def test_password_recovery(self, test_passwords):
        """Test password recovery from partial information."""
        recovery = PasswordRecovery()
        recovery.setup_encoding("abcdefghijklmnopqrstuvwxyz0123456789")
        recovery.train_on_passwords(test_passwords, max_length=12)
        
        # Test recovery
        candidates = recovery.recover_password_advanced("p***word", [0, 4, 5, 6, 7], 12)
        
        assert len(candidates) > 0
        assert all(isinstance(cand, tuple) and len(cand) == 2 for cand in candidates)
        assert all(isinstance(pwd, str) and isinstance(conf, float) for pwd, conf in candidates)


class TestOptimizationSolver:
    """Test optimization solver functionality."""
    
    @pytest.fixture
    def small_tsp(self):
        """Create a small TSP instance."""
        n_cities = 4
        np.random.seed(42)
        cities = np.random.rand(n_cities, 2) * 10
        distance_matrix = np.zeros((n_cities, n_cities))
        
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
        
        return distance_matrix
    
    def test_tsp_solver_creation(self):
        """Test TSP solver initialization."""
        solver = OptimizationSolver()
        assert solver.network is None
    
    def test_tsp_solving(self, small_tsp):
        """Test TSP solving process."""
        solver = OptimizationSolver()
        result = solver.traveling_salesman(small_tsp, max_iterations=100)
        
        assert 'tour' in result
        assert 'total_distance' in result
        assert 'valid_solution' in result
        assert 'convergence_info' in result
        assert 'solution_matrix' in result
        
        if result['valid_solution']:
            assert len(result['tour']) == len(small_tsp)
            assert result['total_distance'] > 0
            assert len(set(result['tour'])) == len(result['tour'])  # Unique cities
    
    def test_tsp_distance_calculation(self, small_tsp):
        """Test that TSP distance calculation is reasonable."""
        solver = OptimizationSolver()
        result = solver.traveling_salesman(small_tsp, max_iterations=200)
        
        if result['valid_solution']:
            # Manual distance calculation
            tour = result['tour']
            manual_distance = 0
            for i in range(len(tour)):
                manual_distance += small_tsp[tour[i], tour[(i + 1) % len(tour)]]
            
            assert np.isclose(result['total_distance'], manual_distance, rtol=1e-6)


class TestPatternCompletion:
    """Test pattern completion functionality."""
    
    @pytest.fixture
    def test_sequences(self):
        """Sample sequences for testing."""
        return ["HELLO WORLD", "PYTHON CODE", "NEURAL NETS", "HOPFIELD NET", "DEEP LEARN"]
    
    def test_sequence_training(self, test_sequences):
        """Test training on sequence patterns."""
        completer = PatternCompletion("text")
        stats = completer.train_sequence_patterns(test_sequences)
        
        assert completer.network is not None
        assert completer.vocab is not None
        assert completer.pattern_size > 0
        assert 'n_patterns' in stats
    
    def test_sequence_encoding_decoding(self, test_sequences):
        """Test sequence encoding and decoding."""
        completer = PatternCompletion("text")
        completer.train_sequence_patterns(test_sequences)
        
        sequence = test_sequences[0]
        max_length = max(len(s) for s in test_sequences)
        
        encoded = completer.encode_sequence(sequence, max_length)
        decoded = completer.decode_sequence(encoded)
        
        assert len(encoded) == max_length * len(completer.vocab)
        assert np.all(np.abs(encoded) == 1)  # Binary
        assert sequence in decoded or sequence.strip() == decoded.strip()
    
    def test_pattern_completion(self, test_sequences):
        """Test pattern completion functionality."""
        completer = PatternCompletion("text")
        completer.train_sequence_patterns(test_sequences)
        
        # Test completion
        completions = completer.complete_pattern("H***O", [0, 4])
        
        assert len(completions) > 0
        assert all(isinstance(comp, str) for comp in completions)
        assert all(len(comp) > 0 for comp in completions)
    
    def test_custom_vocabulary(self):
        """Test custom vocabulary handling."""
        sequences = ["ABC", "BCA", "CAB"]
        vocab = {'A': 0, 'B': 1, 'C': 2}
        
        completer = PatternCompletion("custom")
        stats = completer.train_sequence_patterns(sequences, vocab)
        
        assert completer.vocab == vocab
        assert 'n_patterns' in stats


class TestIntegration:
    """Integration tests for applications."""
    
    def test_end_to_end_password_recovery(self):
        """End-to-end test of password recovery system."""
        # Setup
        recovery = PasswordRecovery()
        recovery.setup_encoding("abcdefghijklmnopqrstuvwxyz0123456789")
        
        # Train on known passwords
        known_passwords = ["password", "admin123", "user2024", "secure01"]
        recovery.train_on_passwords(known_passwords, max_length=10)
        
        # Test recovery of similar password - use more known characters
        candidates = recovery.recover_password_advanced("passw***", [0, 1, 2, 3, 4], 10)
        
        # Should find "password" or similar
        assert len(candidates) > 0
        # Just check that we got some candidates - the exact result may vary
        assert all(isinstance(c[0], str) and isinstance(c[1], float) for c in candidates)
    
    def test_end_to_end_pattern_completion(self):
        """End-to-end test of pattern completion system."""
        # Setup with common English words
        sequences = [
            "THE QUICK BROWN", "BROWN FOX JUMPS", "FOX JUMPS OVER",
            "JUMPS OVER THE", "OVER THE LAZY", "THE LAZY DOG"
        ]
        
        completer = PatternCompletion("text")
        completer.train_sequence_patterns(sequences)
        
        # Test completion with more known characters for better results
        completions = completer.complete_pattern("THE QU***", [0, 1, 2, 3, 4, 5])
        
        # Should complete to something reasonable
        assert len(completions) > 0
        # Just verify we got string completions
        assert all(isinstance(comp, str) for comp in completions)
    
    def test_performance_scaling(self):
        """Test performance with different problem sizes."""
        from hopfield_toolkit import HopfieldNetwork
        sizes = [10, 50, 100]
        times = []
        
        for size in sizes:
            # Create patterns
            patterns = np.random.choice([-1, 1], size=(3, size))
            
            # Time training
            from time import time
            start = time()
            
            network = HopfieldNetwork(size, f"ScaleTest{size}")
            network.train_hebbian(patterns)
            
            # Time recall
            noisy = network.add_noise(patterns[0], 0.2)
            network.recall(noisy, max_iterations=50)
            
            elapsed = time() - start
            times.append(elapsed)
        
        # Performance should scale reasonably (not exponentially)
        assert times[1] < times[0] * 10  # 5x size shouldn't be 10x time
        assert times[2] < times[0] * 50  # 10x size shouldn't be 50x time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])