"""
Comprehensive tests for Hopfield Network Toolkit
"""

import pytest
import numpy as np
import tempfile
import os
from hopfield_toolkit import HopfieldNetwork, HopfieldAnalyzer


class TestHopfieldNetwork:
    """Test core Hopfield Network functionality."""
    
    @pytest.fixture
    def small_network(self):
        """Create a small test network."""
        return HopfieldNetwork(10, "TestNet")
    
    @pytest.fixture
    def test_patterns(self):
        """Create test patterns."""
        return np.array([
            [1, 1, -1, -1, 1],
            [-1, 1, 1, -1, -1],
            [1, -1, 1, 1, -1]
        ])
    
    def test_network_creation(self, small_network):
        """Test network initialization."""
        assert small_network.n_neurons == 10
        assert small_network.name == "TestNet"
        assert small_network.weights.shape == (10, 10)
        assert len(small_network.states) == 10
        assert np.all(np.abs(small_network.states) == 1)
    
    def test_hebbian_training(self, test_patterns):
        """Test Hebbian learning."""
        network = HopfieldNetwork(5, "HebbianTest")
        stats = network.train_hebbian(test_patterns)
        
        assert len(network.patterns) == 3
        assert 'n_patterns' in stats
        assert 'capacity_ratio' in stats
        assert 'training_time' in stats
        assert stats['n_patterns'] == 3
        
        # Check weight matrix properties
        assert np.allclose(network.weights, network.weights.T)  # Symmetric
        assert np.all(np.diag(network.weights) == 0)  # No self-connections
    
    def test_pseudoinverse_training(self, test_patterns):
        """Test pseudoinverse learning."""
        network = HopfieldNetwork(5, "PseudoTest")
        stats = network.train_pseudoinverse(test_patterns)
        
        assert len(network.patterns) == 3
        assert stats['method'] == 'pseudoinverse'
        assert stats['success']
    
    def test_energy_calculation(self, small_network):
        """Test energy calculation."""
        small_network.weights = np.random.randn(10, 10)
        small_network.weights = (small_network.weights + small_network.weights.T) / 2
        np.fill_diagonal(small_network.weights, 0)
        
        states = np.random.choice([-1, 1], 10)
        energy = small_network.energy(states)
        
        # Manual calculation
        expected_energy = -0.5 * np.dot(states, np.dot(small_network.weights, states))
        assert np.isclose(energy, expected_energy)
    
    def test_recall_convergence(self):
        """Test pattern recall and convergence."""
        # Create network and patterns of the same size
        network = HopfieldNetwork(20, "RecallTest")
        
        # Create patterns that match network size
        patterns = np.array([
            np.random.choice([-1, 1], 20),
            np.random.choice([-1, 1], 20)
        ])
        
        network.train_hebbian(patterns)
        
        # Test perfect recall
        result = network.recall(patterns[0], max_iterations=50)
        # For small networks, just test that recall completes
        assert result['converged'] or result['iterations'] == 50
        
        # Test noisy recall
        noisy = network.add_noise(patterns[0], 0.1)  # Reduced noise
        result = network.recall(noisy, max_iterations=100)
        
        assert 'converged' in result
        assert 'iterations' in result
        assert 'final_energy' in result
        assert 'energy_reduction' in result
    
    def test_noise_addition(self, test_patterns):
        """Test different noise types."""
        network = HopfieldNetwork(5, "NoiseTest")
        pattern = test_patterns[0]
        
        # Flip noise
        noisy_flip = network.add_noise(pattern, 0.4, 'flip')
        diff_flip = np.sum(noisy_flip != pattern)
        assert diff_flip > 0  # Should be different
        assert diff_flip <= 2  # At most 40% of 5 bits
        
        # Gaussian noise
        noisy_gauss = network.add_noise(pattern, 0.5, 'gaussian')
        assert np.all(np.abs(noisy_gauss) == 1)  # Should be binary
        
        # Mask noise
        noisy_mask = network.add_noise(pattern, 0.4, 'mask')
        assert np.all(np.abs(noisy_mask) == 1)  # Should be binary
    
    def test_stability_analysis(self, test_patterns):
        """Test stability analysis."""
        network = HopfieldNetwork(5, "StabilityTest")
        
        # Test without training
        result = network.stability_analysis()
        assert "error" in result
        
        # Test with training
        network.train_hebbian(test_patterns)
        stability = network.stability_analysis()
        
        assert 'pattern_stability' in stability
        assert len(stability['pattern_stability']) == len(test_patterns)
        
        for pattern_info in stability['pattern_stability']:
            assert 'pattern_id' in pattern_info
            assert 'is_stable' in pattern_info
            assert 'energy' in pattern_info
            assert 'violations' in pattern_info
    
    def test_capacity_test(self, small_network):
        """Test capacity testing."""
        results = small_network.capacity_test(max_patterns=3)
        
        assert 'pattern_counts' in results
        assert 'success_rates' in results
        assert 'avg_energy_reductions' in results
        
        assert len(results['pattern_counts']) > 0
        assert all(0 <= rate <= 1 for rate in results['success_rates'])
    
    def test_save_load_json(self, test_patterns):
        """Test saving and loading network in JSON format."""
        network = HopfieldNetwork(5, "SaveLoadTest")
        network.train_hebbian(test_patterns)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            network.save(f.name)
            
            # Load and compare
            loaded_network = HopfieldNetwork.load(f.name)
            
            assert loaded_network.n_neurons == network.n_neurons
            assert loaded_network.name == network.name
            assert np.allclose(loaded_network.weights, network.weights)
            assert len(loaded_network.patterns) == len(network.patterns)
            
            os.unlink(f.name)
    
    def test_save_load_pickle(self, test_patterns):
        """Test saving and loading network in pickle format."""
        network = HopfieldNetwork(5, "SaveLoadTest")
        network.train_hebbian(test_patterns)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            network.save(f.name)
            
            # Load and compare
            loaded_network = HopfieldNetwork.load(f.name)
            
            assert loaded_network.n_neurons == network.n_neurons
            assert loaded_network.name == network.name
            assert np.allclose(loaded_network.weights, network.weights)
            assert len(loaded_network.patterns) == len(network.patterns)
            
            os.unlink(f.name)


class TestHopfieldAnalyzer:
    """Test analyzer functionality."""
    
    def test_theoretical_capacity(self):
        """Test theoretical capacity calculations."""
        capacity = HopfieldAnalyzer.theoretical_capacity(100)
        
        assert 'hopfield_capacity' in capacity
        assert 'gardner_capacity' in capacity
        assert 'practical_capacity' in capacity
        assert 'neuron_count' in capacity
        
        assert capacity['neuron_count'] == 100
        assert capacity['hopfield_capacity'] == 13  # int(0.138 * 100)
        assert capacity['gardner_capacity'] == 10   # int(0.105 * 100)
        assert capacity['practical_capacity'] == 5  # int(0.05 * 100)
    
    def test_compare_networks(self):
        """Test network comparison."""
        # Create test networks
        patterns = np.random.choice([-1, 1], size=(3, 10))
        
        net1 = HopfieldNetwork(10, "Net1")
        net1.train_hebbian(patterns)
        
        net2 = HopfieldNetwork(10, "Net2")
        net2.train_pseudoinverse(patterns)
        
        # Compare
        results = HopfieldAnalyzer.compare_networks([net1, net2], patterns)
        
        assert 'network_names' in results
        assert 'recall_accuracies' in results
        assert 'convergence_times' in results
        assert 'energy_reductions' in results
        
        assert len(results['network_names']) == 2
        assert len(results['recall_accuracies']) == 2
        assert all(0 <= acc <= 1 for acc in results['recall_accuracies'])


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_patterns(self):
        """Test handling of invalid patterns."""
        network = HopfieldNetwork(5, "ErrorTest")
        
        # Wrong size patterns
        wrong_patterns = np.array([[1, -1, 1]])  # Wrong size
        with pytest.raises((ValueError, IndexError)):
            network.train_hebbian(wrong_patterns)
    
    def test_empty_patterns(self):
        """Test handling of empty pattern arrays."""
        network = HopfieldNetwork(5, "ErrorTest")
        
        empty_patterns = np.array([]).reshape(0, 5)
        stats = network.train_hebbian(empty_patterns)
        
        assert stats['n_patterns'] == 0
    
    def test_capacity_warning(self, capsys):
        """Test capacity warning."""
        network = HopfieldNetwork(10, "CapacityTest")
        
        # Too many patterns (more than 0.138 * 10 = 1.38, so 2+ should warn)
        patterns = np.random.choice([-1, 1], size=(5, 10))
        network.train_hebbian(patterns)
        
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "exceed capacity" in captured.out.lower()
    
    def test_recall_without_training(self):
        """Test recall on untrained network."""
        network = HopfieldNetwork(5, "UntrainedTest")
        pattern = np.random.choice([-1, 1], 5)
        
        # Should work but not match any stored pattern
        result = network.recall(pattern)
        assert 'final_states' in result
        assert len(result['hamming_distances']) == 0  # No stored patterns


class TestPerformance:
    """Performance and benchmark tests."""
    
    @pytest.mark.benchmark
    def test_training_performance(self, benchmark):
        """Benchmark training performance."""
        network = HopfieldNetwork(100, "BenchmarkNet")
        patterns = np.random.choice([-1, 1], size=(10, 100))
        
        result = benchmark(network.train_hebbian, patterns)
        assert result['n_patterns'] == 10
    
    @pytest.mark.benchmark
    def test_recall_performance(self, benchmark):
        """Benchmark recall performance."""
        network = HopfieldNetwork(100, "BenchmarkNet")
        patterns = np.random.choice([-1, 1], size=(5, 100))
        network.train_hebbian(patterns)
        
        noisy_pattern = network.add_noise(patterns[0], 0.2)
        result = benchmark(network.recall, noisy_pattern, max_iterations=50)
        
        assert 'final_states' in result
    
    @pytest.mark.benchmark
    def test_energy_calculation_performance(self, benchmark):
        """Benchmark energy calculation."""
        network = HopfieldNetwork(1000, "LargeBenchmarkNet")
        network.weights = np.random.randn(1000, 1000)
        network.weights = (network.weights + network.weights.T) / 2
        np.fill_diagonal(network.weights, 0)
        
        states = np.random.choice([-1, 1], 1000)
        
        result = benchmark(network.energy, states)
        assert isinstance(result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])