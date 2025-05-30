"""
Real-world applications of Hopfield Networks
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
from hopfield_toolkit import HopfieldNetwork
import json
import os


class ImageRestoration:
    """Image denoising and restoration using Hopfield Networks."""
    
    def __init__(self, patch_size: int = 8):
        self.patch_size = patch_size
        self.networks = {}
        self.patch_patterns = {}
        
    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract overlapping patches from image."""
        h, w = image.shape
        patches = []
        
        for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch.flatten())
                
        return np.array(patches)
    
    def train_on_images(self, clean_images: List[np.ndarray], 
                       image_type: str = "general") -> Dict:
        """Train network on clean image patches."""
        all_patches = []
        
        for img in clean_images:
            # Convert to binary
            binary_img = np.where(img > 128, 1, -1)
            patches = self.extract_patches(binary_img)
            all_patches.extend(patches)
            
        all_patches = np.array(all_patches)
        
        # Remove duplicate patches
        unique_patches = np.unique(all_patches, axis=0)
        
        # Create network
        network = HopfieldNetwork(self.patch_size ** 2, f"ImageNet_{image_type}")
        training_stats = network.train_hebbian(unique_patches[:50])  # Limit for capacity
        
        self.networks[image_type] = network
        self.patch_patterns[image_type] = unique_patches[:50]
        
        return training_stats
    
    def restore_image(self, noisy_image: np.ndarray, 
                     image_type: str = "general") -> np.ndarray:
        """Restore noisy image using trained network."""
        if image_type not in self.networks:
            raise ValueError(f"No network trained for type: {image_type}")
            
        network = self.networks[image_type]
        binary_img = np.where(noisy_image > 128, 1, -1)
        h, w = binary_img.shape
        
        restored = np.zeros_like(binary_img)
        overlap_count = np.zeros_like(binary_img)
        
        # Process patches
        for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                patch = binary_img[i:i+self.patch_size, j:j+self.patch_size]
                
                # Restore patch
                result = network.recall(patch.flatten(), max_iterations=50)
                restored_patch = result['final_states'].reshape(self.patch_size, self.patch_size)
                
                # Add to result with overlap handling
                restored[i:i+self.patch_size, j:j+self.patch_size] += restored_patch
                overlap_count[i:i+self.patch_size, j:j+self.patch_size] += 1
                
        # Average overlapping regions
        restored = restored / np.maximum(overlap_count, 1)
        restored = np.where(restored > 0, 255, 0)
        
        return restored.astype(np.uint8)
    
    def train_on_image(self, image: np.ndarray, image_type: str = "general") -> Dict:
        """Convenience method to train on a single image."""
        return self.train_on_images([image], image_type)


class PasswordRecovery:
    """Password pattern completion using Hopfield Networks."""
    
    def __init__(self):
        self.network = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.pattern_length = 0
        
    def setup_encoding(self, charset: str = "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"):
        """Setup character encoding."""
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(charset)}
        
    def encode_password(self, password: str, length: int) -> np.ndarray:
        """Encode password as binary pattern."""
        # Pad or truncate to fixed length
        padded = (password + ' ' * length)[:length]
        
        # One-hot encoding for each position
        pattern = []
        for char in padded:
            char_vec = np.zeros(len(self.char_to_idx))
            if char in self.char_to_idx:
                char_vec[self.char_to_idx[char]] = 1
            pattern.extend(char_vec)
            
        # Convert to bipolar
        return np.where(np.array(pattern) == 1, 1, -1)
    
    def decode_pattern(self, pattern: np.ndarray) -> str:
        """Decode binary pattern back to password."""
        chars_per_pos = len(self.char_to_idx)
        password = ""
        
        for i in range(0, len(pattern), chars_per_pos):
            char_vec = pattern[i:i+chars_per_pos]
            # Find most activated character
            if np.any(char_vec > 0):
                char_idx = np.argmax(char_vec)
                if char_idx in self.idx_to_char:
                    password += self.idx_to_char[char_idx]
                    
        return password.strip()
    
    def train_on_passwords(self, passwords: List[str], max_length: int = 12) -> Dict:
        """Train network on common password patterns."""
        self.pattern_length = max_length * len(self.char_to_idx)
        
        # Encode all passwords
        patterns = []
        for pwd in passwords:
            if len(pwd) <= max_length:
                pattern = self.encode_password(pwd, max_length)
                patterns.append(pattern)
                
        patterns = np.array(patterns)
        
        # Train network
        self.network = HopfieldNetwork(self.pattern_length, "PasswordNet")
        return self.network.train_hebbian(patterns)
    
    def recover_password_advanced(self, partial_password: str, known_positions: List[int], 
                        max_length: int = 12) -> List[Tuple[str, float]]:
        """Recover password from partial information."""
        if self.network is None:
            raise ValueError("Network not trained")
            
        # Use the trained network's expected length
        trained_length = self.pattern_length // len(self.char_to_idx)
        # Always use the trained length to match network size
        effective_length = trained_length
            
        # Create corrupted pattern
        corrupted = self.encode_password(' ' * effective_length, effective_length)
        
        # Set known characters
        chars_per_pos = len(self.char_to_idx)
        for pos in known_positions:
            if pos < len(partial_password) and pos < effective_length:
                char = partial_password[pos]
                if char in self.char_to_idx:
                    start_idx = pos * chars_per_pos
                    corrupted[start_idx:start_idx + chars_per_pos] = -1
                    corrupted[start_idx + self.char_to_idx[char]] = 1
                    
        # Recall multiple times for different solutions
        candidates = []
        for _ in range(10):
            # Add small random noise for variation
            noisy = self.network.add_noise(corrupted, 0.05)
            result = self.network.recall(noisy, max_iterations=100)
            
            decoded = self.decode_pattern(result['final_states'])
            confidence = 1.0 - result['energy_reduction']  # Higher energy reduction = lower confidence
            
            candidates.append((decoded, confidence))
            
        # Remove duplicates and sort by confidence
        unique_candidates = list(set(candidates))
        return sorted(unique_candidates, key=lambda x: x[1], reverse=True)
    
    def recover_password(self, corrupted_password: str) -> str:
        """Simple interface to recover password with _ placeholders."""
        if self.network is None:
            raise ValueError("Network not trained")
        
        # Find known positions (not _)
        known_positions = [i for i, char in enumerate(corrupted_password) if char != '_']
        
        # Use the advanced recover method
        candidates = self.recover_password_advanced(corrupted_password, known_positions, len(corrupted_password))
        
        # Return the best candidate
        return candidates[0][0] if candidates else corrupted_password
    
    def _recover_password_full_backup(self, partial_password: str, known_positions: List[int], 
                        max_length: int = 12) -> List[Tuple[str, float]]:
        """Recover password from partial information (original method)."""
        if self.network is None:
            raise ValueError("Network not trained")
            
        # Create corrupted pattern
        corrupted = self.encode_password(' ' * max_length, max_length)
        
        # Set known characters
        chars_per_pos = len(self.char_to_idx)
        for pos in known_positions:
            if pos < len(partial_password) and pos < effective_length:
                char = partial_password[pos]
                if char in self.char_to_idx:
                    start_idx = pos * chars_per_pos
                    corrupted[start_idx:start_idx + chars_per_pos] = -1
                    corrupted[start_idx + self.char_to_idx[char]] = 1
                    
        # Recall multiple times for different solutions
        candidates = []
        for _ in range(10):
            # Add small random noise for variation
            noisy = self.network.add_noise(corrupted, 0.05)
            result = self.network.recall(noisy, max_iterations=100)
            
            decoded = self.decode_pattern(result['final_states'])
            confidence = 1.0 - result['energy_reduction']  # Higher energy reduction = lower confidence
            
            candidates.append((decoded, confidence))
            
        # Remove duplicates and sort by confidence
        unique_candidates = list(set(candidates))
        return sorted(unique_candidates, key=lambda x: x[1], reverse=True)


class OptimizationSolver:
    """Solve optimization problems using Hopfield Networks."""
    
    def __init__(self):
        self.network = None
        
    def traveling_salesman(self, distance_matrix: np.ndarray, 
                          max_iterations: int = 1000) -> Dict:
        """Solve TSP using Hopfield Network."""
        n_cities = len(distance_matrix)
        n_neurons = n_cities * n_cities
        
        # Create network
        self.network = HopfieldNetwork(n_neurons, "TSP_Solver")
        
        # Build weights based on TSP constraints
        weights = np.zeros((n_neurons, n_neurons))
        
        # Constraint weights
        A = 500  # One city per position
        B = 500  # One position per city  
        C = 200  # Distance minimization
        D = 500  # Valid tour constraint
        
        for i in range(n_cities):
            for j in range(n_cities):
                neuron_ij = i * n_cities + j
                
                # A: Each city appears exactly once
                for k in range(n_cities):
                    if k != j:
                        neuron_ik = i * n_cities + k
                        weights[neuron_ij, neuron_ik] -= A
                        
                # B: Each position has exactly one city
                for k in range(n_cities):
                    if k != i:
                        neuron_kj = k * n_cities + j
                        weights[neuron_ij, neuron_kj] -= B
                        
                # C: Distance minimization
                for k in range(n_cities):
                    if k != i:
                        next_pos = (j + 1) % n_cities
                        neuron_k_next = k * n_cities + next_pos
                        weights[neuron_ij, neuron_k_next] -= C * distance_matrix[i, k]
                        
        self.network.weights = weights
        
        # Random initial state
        self.network.states = np.random.choice([-1, 1], size=n_neurons)
        
        # Solve
        result = self.network.recall(self.network.states, max_iterations=max_iterations)
        
        # Decode solution
        solution_matrix = result['final_states'].reshape(n_cities, n_cities)
        solution_matrix = np.where(solution_matrix > 0, 1, 0)
        
        # Extract tour
        tour = []
        total_distance = 0
        valid_solution = True
        
        try:
            for pos in range(n_cities):
                cities_at_pos = np.where(solution_matrix[:, pos] == 1)[0]
                if len(cities_at_pos) == 1:
                    tour.append(cities_at_pos[0])
                else:
                    valid_solution = False
                    break
                    
            if valid_solution and len(tour) == n_cities:
                for i in range(n_cities):
                    total_distance += distance_matrix[tour[i], tour[(i + 1) % n_cities]]
            else:
                valid_solution = False
                
        except:
            valid_solution = False
            
        return {
            'tour': tour if valid_solution else [],
            'total_distance': total_distance if valid_solution else float('inf'),
            'valid_solution': valid_solution,
            'convergence_info': result,
            'solution_matrix': solution_matrix
        }


class PatternCompletion:
    """Complete partial patterns in various domains."""
    
    def __init__(self, pattern_type: str = "binary"):
        self.pattern_type = pattern_type
        self.network = None
        self.pattern_size = 0
        
    def train_sequence_patterns(self, sequences: List[str], 
                               vocab: Optional[Dict] = None) -> Dict:
        """Train on sequence patterns (text, DNA, etc.)."""
        if vocab is None:
            # Build vocabulary
            all_chars = set(''.join(sequences))
            vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}
            
        self.vocab = vocab
        self.pattern_size = max(len(seq) for seq in sequences) * len(vocab)
        
        # Encode sequences
        patterns = []
        for seq in sequences:
            pattern = self.encode_sequence(seq, max(len(s) for s in sequences))
            patterns.append(pattern)
            
        patterns = np.array(patterns)
        
        # Train network
        self.network = HopfieldNetwork(self.pattern_size, f"SeqNet_{self.pattern_type}")
        return self.network.train_hebbian(patterns)
    
    def encode_sequence(self, sequence: str, max_length: int) -> np.ndarray:
        """Encode sequence as binary pattern."""
        # Pad sequence
        padded = (sequence + ' ' * max_length)[:max_length]
        
        # One-hot encode
        pattern = []
        for char in padded:
            char_vec = np.zeros(len(self.vocab))
            if char in self.vocab:
                char_vec[self.vocab[char]] = 1
            pattern.extend(char_vec)
            
        return np.where(np.array(pattern) == 1, 1, -1)
    
    def decode_sequence(self, pattern: np.ndarray) -> str:
        """Decode pattern back to sequence."""
        vocab_size = len(self.vocab)
        idx_to_char = {idx: char for char, idx in self.vocab.items()}
        
        sequence = ""
        for i in range(0, len(pattern), vocab_size):
            char_vec = pattern[i:i+vocab_size]
            if np.any(char_vec > 0):
                char_idx = np.argmax(char_vec)
                if char_idx in idx_to_char:
                    sequence += idx_to_char[char_idx]
                    
        return sequence.rstrip()  # Only strip trailing spaces
    
    def complete_pattern(self, partial_sequence: str, 
                        known_positions: List[int]) -> List[str]:
        """Complete partial sequence."""
        if self.network is None:
            raise ValueError("Network not trained")
            
        max_length = self.pattern_size // len(self.vocab)
        
        # Create masked pattern
        corrupted = np.random.choice([-1, 1], size=self.pattern_size)
        
        # Set known positions
        vocab_size = len(self.vocab)
        for pos in known_positions:
            if pos < len(partial_sequence):
                char = partial_sequence[pos]
                if char in self.vocab:
                    start_idx = pos * vocab_size
                    corrupted[start_idx:start_idx + vocab_size] = -1
                    corrupted[start_idx + self.vocab[char]] = 1
                    
        # Generate completions
        completions = []
        for _ in range(5):
            noisy = self.network.add_noise(corrupted, 0.1)
            result = self.network.recall(noisy, max_iterations=100)
            completion = self.decode_sequence(result['final_states'])
            if completion not in completions:
                completions.append(completion)
                
        return completions
    
    def complete_sequence(self, partial_sequence: str) -> str:
        """Simple interface to complete a sequence with _ placeholders."""
        if self.network is None:
            raise ValueError("Network not trained")
        
        # Get the max length from training
        max_length = self.pattern_size // len(self.vocab)
        
        # Create a proper input pattern by padding/truncating
        padded_partial = (partial_sequence + ' ' * max_length)[:max_length]
        
        # Create corrupted pattern for unknown positions
        corrupted = np.random.choice([-1, 1], size=self.pattern_size)
        
        # Set known characters
        vocab_size = len(self.vocab)
        for i, char in enumerate(padded_partial):
            if char != '_' and char in self.vocab:
                start_idx = i * vocab_size
                corrupted[start_idx:start_idx + vocab_size] = -1
                corrupted[start_idx + self.vocab[char]] = 1
        
        # Recall to get completion
        result = self.network.recall(corrupted, max_iterations=50)
        completed = self.decode_sequence(result['final_states'])
        
        # Ensure the result is the same length as the input
        result = completed + ' ' * len(partial_sequence)  # Pad if needed
        return result[:len(partial_sequence)]  # Truncate to input length


def create_demo_data():
    """Create demo data for testing applications."""
    # Create simple test images
    images = []
    for i in range(5):
        img = np.random.randint(0, 256, (64, 64))
        images.append(img)
        
    # Create password list
    passwords = [
        "password123", "admin2024", "user1234", "test123", 
        "secure99", "login2024", "access01", "system2024"
    ]
    
    # Create TSP instance
    n_cities = 5
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2) * 100
    distance_matrix = np.zeros((n_cities, n_cities))
    
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
                
    # Text sequences
    sequences = [
        "HELLO WORLD", "PYTHON CODE", "NEURAL NETS", 
        "HOPFIELD NET", "DEEP LEARN", "AI SYSTEMS"
    ]
    
    return {
        'images': images,
        'passwords': passwords,
        'distance_matrix': distance_matrix,
        'sequences': sequences
    }


if __name__ == "__main__":
    print("Hopfield Network Applications Demo")
    print("=" * 50)
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Test Image Restoration
    print("\n1. Image Restoration")
    print("-" * 20)
    img_restorer = ImageRestoration(patch_size=4)
    img_restorer.train_on_images(demo_data['images'][:3])
    print("Image restoration network trained")
    
    # Test Password Recovery
    print("\n2. Password Recovery")
    print("-" * 20)
    pwd_recovery = PasswordRecovery()
    pwd_recovery.setup_encoding()
    pwd_recovery.train_on_passwords(demo_data['passwords'])
    candidates = pwd_recovery.recover_password("p***word", [0, 4, 5, 6, 7], 12)
    print(f"Password candidates: {candidates[:3]}")
    
    # Test TSP Solver
    print("\n3. Traveling Salesman Problem")
    print("-" * 30)
    tsp_solver = OptimizationSolver()
    result = tsp_solver.traveling_salesman(demo_data['distance_matrix'])
    print(f"TSP Solution valid: {result['valid_solution']}")
    if result['valid_solution']:
        print(f"Tour: {result['tour']}")
        print(f"Distance: {result['total_distance']:.2f}")
    
    # Test Pattern Completion
    print("\n4. Pattern Completion")
    print("-" * 20)
    pattern_completer = PatternCompletion()
    pattern_completer.train_sequence_patterns(demo_data['sequences'])
    completions = pattern_completer.complete_pattern("H***O W****", [0, 4, 6])
    print(f"Completions: {completions}")