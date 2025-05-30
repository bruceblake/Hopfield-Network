"""
Hopfield Network Toolkit - Professional implementation for real-world applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HopfieldNetwork:
    """Enhanced Hopfield Network with professional features."""
    
    def __init__(self, n_neurons: int, name: str = "HopfieldNet"):
        self.n_neurons = n_neurons
        self.name = name
        self.weights = np.zeros((n_neurons, n_neurons))
        self.states = np.random.choice([-1, 1], size=n_neurons)
        self.patterns = []
        self.energy_history = []
        self.training_info = {}
        self.convergence_stats = {}
        
    def train_hebbian(self, patterns: np.ndarray, normalize: bool = True) -> Dict:
        """Train using Hebbian learning with detailed statistics."""
        start_time = datetime.now()
        n_patterns = patterns.shape[0]
        
        if n_patterns > 0.138 * self.n_neurons:
            print(f"Warning: {n_patterns} patterns may exceed capacity (~{int(0.138 * self.n_neurons)})")
            
        self.patterns = patterns.copy()
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Hebbian learning
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
            
        if normalize:
            self.weights /= n_patterns
            
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
        # Calculate training statistics
        self.training_info = {
            'n_patterns': n_patterns,
            'capacity_ratio': n_patterns / (0.138 * self.n_neurons),
            'weight_stats': {
                'mean': float(np.mean(self.weights)),
                'std': float(np.std(self.weights)),
                'max': float(np.max(self.weights)),
                'min': float(np.min(self.weights))
            },
            'training_time': (datetime.now() - start_time).total_seconds(),
            'normalized': normalize
        }
        
        return self.training_info
    
    def train_pseudoinverse(self, patterns: np.ndarray) -> Dict:
        """Train using pseudoinverse rule (better capacity)."""
        start_time = datetime.now()
        self.patterns = patterns.copy()
        
        try:
            # Pseudoinverse learning rule
            P = patterns.T
            self.weights = P @ np.linalg.pinv(P)
            np.fill_diagonal(self.weights, 0)
            
            self.training_info = {
                'method': 'pseudoinverse',
                'n_patterns': patterns.shape[0],
                'training_time': (datetime.now() - start_time).total_seconds(),
                'success': True
            }
        except np.linalg.LinAlgError:
            print("Pseudoinverse failed, falling back to Hebbian")
            return self.train_hebbian(patterns)
            
        return self.training_info
    
    def energy(self, states: Optional[np.ndarray] = None) -> float:
        """Calculate network energy."""
        if states is None:
            states = self.states
        return -0.5 * np.dot(states, np.dot(self.weights, states))
    
    def local_field(self, neuron_idx: int, states: Optional[np.ndarray] = None) -> float:
        """Calculate local field for a specific neuron."""
        if states is None:
            states = self.states
        return np.dot(self.weights[neuron_idx], states)
    
    def stability_analysis(self) -> Dict:
        """Analyze network stability and attractors."""
        if len(self.patterns) == 0:
            return {"error": "No patterns trained"}
            
        results = {
            'pattern_stability': [],
            'spurious_attractors': 0,
            'basin_sizes': []
        }
        
        # Test each stored pattern
        for i, pattern in enumerate(self.patterns):
            self.states = pattern.copy()
            initial_energy = self.energy()
            
            # Check if it's a fixed point
            local_fields = np.dot(self.weights, pattern)
            is_stable = np.all(np.sign(local_fields) == pattern)
            
            results['pattern_stability'].append({
                'pattern_id': i,
                'is_stable': bool(is_stable),
                'energy': float(initial_energy),
                'violations': int(np.sum(np.sign(local_fields) != pattern))
            })
            
        return results
    
    def recall(self, input_pattern: np.ndarray, max_iterations: int = 100, 
               async_update: bool = True, track_energy: bool = True) -> Dict:
        """Enhanced recall with detailed tracking."""
        self.states = input_pattern.copy()
        initial_energy = self.energy()
        
        if track_energy:
            self.energy_history = [initial_energy]
            
        convergence_info = {
            'converged': False,
            'iterations': 0,
            'final_energy': initial_energy,
            'energy_reduction': 0,
            'hamming_distances': [],
            'pattern_matches': []
        }
        
        for iteration in range(max_iterations):
            old_states = self.states.copy()
            
            if async_update:
                # Asynchronous update
                indices = np.random.permutation(self.n_neurons)
                changed = False
                
                for i in indices:
                    local_field = self.local_field(i)
                    new_state = 1 if local_field > 0 else -1
                    
                    if new_state != self.states[i]:
                        self.states[i] = new_state
                        changed = True
                        
                if not changed:
                    convergence_info['converged'] = True
                    break
            else:
                # Synchronous update
                local_fields = np.dot(self.weights, self.states)
                new_states = np.where(local_fields > 0, 1, -1)
                
                if np.array_equal(new_states, self.states):
                    convergence_info['converged'] = True
                    break
                    
                self.states = new_states
                
            if track_energy:
                self.energy_history.append(self.energy())
                
        # Final analysis
        convergence_info.update({
            'iterations': iteration + 1,
            'final_energy': float(self.energy()),
            'energy_reduction': float(initial_energy - self.energy()),
            'final_states': self.states.copy()
        })
        
        # Check which stored pattern it matches
        if len(self.patterns) > 0:
            for i, pattern in enumerate(self.patterns):
                hamming_dist = np.sum(self.states != pattern) / len(pattern)
                convergence_info['hamming_distances'].append(hamming_dist)
                
                if hamming_dist == 0:
                    convergence_info['pattern_matches'].append(i)
                    
        return convergence_info
    
    def add_noise(self, pattern: np.ndarray, noise_level: float, 
                  noise_type: str = 'flip') -> np.ndarray:
        """Add different types of noise to patterns."""
        noisy = pattern.copy()
        n_neurons = len(pattern)
        
        if noise_type == 'flip':
            # Flip random bits
            n_flip = int(noise_level * n_neurons)
            flip_indices = np.random.choice(n_neurons, n_flip, replace=False)
            noisy[flip_indices] *= -1
            
        elif noise_type == 'gaussian':
            # Add Gaussian noise and threshold
            noise = np.random.normal(0, noise_level, n_neurons)
            noisy = np.sign(pattern + noise)
            noisy[noisy == 0] = 1  # Handle zero values
            
        elif noise_type == 'mask':
            # Set random neurons to random values
            n_mask = int(noise_level * n_neurons)
            mask_indices = np.random.choice(n_neurons, n_mask, replace=False)
            noisy[mask_indices] = np.random.choice([-1, 1], n_mask)
            
        return noisy
    
    def capacity_test(self, max_patterns: Optional[int] = None) -> Dict:
        """Test network capacity with random patterns."""
        if max_patterns is None:
            max_patterns = int(0.5 * self.n_neurons)
            
        results = {
            'pattern_counts': [],
            'success_rates': [],
            'avg_energy_reductions': []
        }
        
        for n_patterns in range(1, max_patterns + 1):
            # Generate random patterns
            patterns = np.random.choice([-1, 1], size=(n_patterns, self.n_neurons))
            self.train_hebbian(patterns)
            
            # Test recall
            successes = 0
            energy_reductions = []
            
            for pattern in patterns:
                noisy = self.add_noise(pattern, 0.1)
                result = self.recall(noisy, max_iterations=50, track_energy=False)
                
                if 0 in result.get('pattern_matches', []):
                    successes += 1
                    
                energy_reductions.append(result['energy_reduction'])
                
            success_rate = successes / n_patterns
            
            results['pattern_counts'].append(n_patterns)
            results['success_rates'].append(success_rate)
            results['avg_energy_reductions'].append(np.mean(energy_reductions))
            
            # Stop if success rate drops below threshold
            if success_rate < 0.5:
                break
                
        return results
    
    def visualize_weights(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize weight matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(self.weights, cmap='RdBu_r', center=0, 
                   square=True, cbar_kws={'label': 'Weight'})
        plt.title(f'{self.name} - Weight Matrix')
        plt.xlabel('Neuron j')
        plt.ylabel('Neuron i')
        plt.tight_layout()
        plt.show()
    
    def visualize_energy_landscape(self, pattern_shape: Optional[Tuple] = None) -> None:
        """Visualize energy over iterations."""
        if not self.energy_history:
            print("No energy history available. Run recall with track_energy=True first.")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.energy_history, 'b-', marker='o', linewidth=2, markersize=4)
        plt.title(f'{self.name} - Energy Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save network to file."""
        data = {
            'n_neurons': self.n_neurons,
            'name': self.name,
            'weights': self.weights,
            'patterns': self.patterns,
            'training_info': self.training_info,
            'convergence_stats': self.convergence_stats
        }
        
        if filepath.endswith('.json'):
            # JSON format (less efficient but readable)
            data['weights'] = self.weights.tolist()
            data['patterns'] = [p.tolist() for p in self.patterns]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Pickle format (efficient)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
    @classmethod
    def load(cls, filepath: str) -> 'HopfieldNetwork':
        """Load network from file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            data['weights'] = np.array(data['weights'])
            data['patterns'] = [np.array(p) for p in data['patterns']]
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
        network = cls(data['n_neurons'], data['name'])
        network.weights = data['weights']
        network.patterns = data['patterns']
        network.training_info = data.get('training_info', {})
        network.convergence_stats = data.get('convergence_stats', {})
        
        return network


class HopfieldAnalyzer:
    """Advanced analysis tools for Hopfield Networks."""
    
    @staticmethod
    def compare_networks(networks: List[HopfieldNetwork], 
                        test_patterns: np.ndarray) -> Dict:
        """Compare multiple networks on same test data."""
        results = {
            'network_names': [net.name for net in networks],
            'recall_accuracies': [],
            'convergence_times': [],
            'energy_reductions': []
        }
        
        for network in networks:
            accuracies = []
            times = []
            energy_reds = []
            
            for pattern in test_patterns:
                noisy = network.add_noise(pattern, 0.2)
                result = network.recall(noisy, max_iterations=100)
                
                # Check accuracy
                best_match_idx = np.argmin([
                    np.sum(result['final_states'] != p) 
                    for p in network.patterns
                ])
                accuracy = 1 - (np.sum(result['final_states'] != network.patterns[best_match_idx]) / len(pattern))
                
                accuracies.append(accuracy)
                times.append(result['iterations'])
                energy_reds.append(result['energy_reduction'])
                
            results['recall_accuracies'].append(np.mean(accuracies))
            results['convergence_times'].append(np.mean(times))
            results['energy_reductions'].append(np.mean(energy_reds))
            
        return results
    
    @staticmethod
    def theoretical_capacity(n_neurons: int) -> Dict:
        """Calculate theoretical capacity limits."""
        return {
            'hopfield_capacity': int(0.138 * n_neurons),
            'gardner_capacity': int(0.105 * n_neurons),  # With high probability
            'practical_capacity': int(0.05 * n_neurons),  # Conservative estimate
            'neuron_count': n_neurons
        }


if __name__ == "__main__":
    # Quick demo
    print("Hopfield Network Toolkit Demo")
    print("=" * 40)
    
    # Create network
    net = HopfieldNetwork(100, "Demo Network")
    
    # Generate test patterns
    patterns = np.random.choice([-1, 1], size=(5, 100))
    
    # Train and analyze
    training_stats = net.train_hebbian(patterns)
    print(f"Trained on {training_stats['n_patterns']} patterns")
    print(f"Capacity ratio: {training_stats['capacity_ratio']:.2f}")
    
    # Test recall
    noisy = net.add_noise(patterns[0], 0.3)
    result = net.recall(noisy)
    print(f"Recall converged: {result['converged']}")
    print(f"Energy reduction: {result['energy_reduction']:.2f}")
    
    # Stability analysis
    stability = net.stability_analysis()
    stable_patterns = sum(1 for p in stability['pattern_stability'] if p['is_stable'])
    print(f"Stable patterns: {stable_patterns}/{len(patterns)}")