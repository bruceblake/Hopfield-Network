"""
Practical applications of Hopfield Networks for real-world use cases.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
import hashlib
from hopfield_network import HopfieldNetwork


class ImageAssociativeMemory:
    """Store and retrieve images using Hopfield Networks."""
    
    def __init__(self, image_size: Tuple[int, int] = (32, 32)):
        self.image_size = image_size
        self.n_pixels = image_size[0] * image_size[1]
        self.network = HopfieldNetwork(self.n_pixels)
        self.stored_images = {}
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image to binary pattern."""
        img = Image.open(image_path).convert('L')
        img = img.resize(self.image_size)
        img_array = np.array(img)
        # Convert to binary (-1, 1)
        threshold = np.mean(img_array)
        binary = np.where(img_array > threshold, 1, -1)
        return binary.flatten()
    
    def store_image(self, image_path: str, label: str):
        """Store an image pattern with a label."""
        pattern = self.preprocess_image(image_path)
        image_hash = hashlib.md5(pattern.tobytes()).hexdigest()
        self.stored_images[image_hash] = {
            'label': label,
            'pattern': pattern,
            'path': image_path
        }
        
    def train_network(self):
        """Train the network with all stored images."""
        if not self.stored_images:
            raise ValueError("No images stored")
        
        patterns = np.array([img['pattern'] for img in self.stored_images.values()])
        self.network.train(patterns)
        
    def recall_image(self, partial_image_path: str, noise_level: float = 0.0) -> Dict:
        """Recall a complete image from partial/noisy input."""
        input_pattern = self.preprocess_image(partial_image_path)
        
        if noise_level > 0:
            input_pattern = self.network.add_noise(input_pattern, noise_level)
            
        recalled_pattern = self.network.recall(input_pattern)
        
        # Find best match
        best_match = None
        best_similarity = -1
        
        for img_hash, img_data in self.stored_images.items():
            similarity = np.mean(recalled_pattern == img_data['pattern'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = img_data
                
        return {
            'recalled_pattern': recalled_pattern.reshape(self.image_size),
            'best_match': best_match,
            'similarity': best_similarity
        }


class PasswordRecoverySystem:
    """Use Hopfield Networks for secure password pattern recovery."""
    
    def __init__(self, pattern_length: int = 64):
        self.pattern_length = pattern_length
        self.network = HopfieldNetwork(pattern_length)
        self.stored_patterns = {}
        
    def create_password_pattern(self, password: str) -> np.ndarray:
        """Convert password to binary pattern using secure hashing."""
        # Create a deterministic pattern from password
        hash_obj = hashlib.sha256(password.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to binary pattern
        pattern = []
        for byte in hash_bytes[:self.pattern_length // 8]:
            for bit in range(8):
                pattern.append(1 if byte & (1 << bit) else -1)
                
        return np.array(pattern[:self.pattern_length])
    
    def store_password_hint(self, user_id: str, password: str, hint: str):
        """Store password pattern with hint."""
        pattern = self.create_password_pattern(password)
        self.stored_patterns[user_id] = {
            'pattern': pattern,
            'hint': hint,
            'hash': hashlib.sha256(password.encode()).hexdigest()
        }
        
    def train_recovery_network(self):
        """Train network with stored password patterns."""
        if not self.stored_patterns:
            raise ValueError("No password patterns stored")
            
        patterns = np.array([data['pattern'] for data in self.stored_patterns.values()])
        self.network.train(patterns)
        
    def recover_password_pattern(self, partial_input: str, user_id: str) -> Dict:
        """Attempt to recover password pattern from partial input."""
        partial_pattern = self.create_password_pattern(partial_input)
        
        # Add controlled noise to simulate partial memory
        noisy_pattern = self.network.add_noise(partial_pattern, 0.3)
        
        # Recall complete pattern
        recalled_pattern = self.network.recall(noisy_pattern)
        
        # Verify against stored pattern
        stored_data = self.stored_patterns.get(user_id)
        if stored_data:
            similarity = np.mean(recalled_pattern == stored_data['pattern'])
            return {
                'success': similarity > 0.85,
                'similarity': similarity,
                'hint': stored_data['hint'],
                'pattern_match': similarity > 0.85
            }
        
        return {'success': False, 'similarity': 0.0}


class OptimizationSolver:
    """Use Hopfield Networks for combinatorial optimization problems."""
    
    def __init__(self, problem_size: int):
        self.problem_size = problem_size
        self.network = None
        
    def solve_tsp(self, distance_matrix: np.ndarray) -> List[int]:
        """Solve Traveling Salesman Problem using Hopfield Network."""
        n_cities = len(distance_matrix)
        n_neurons = n_cities * n_cities
        self.network = HopfieldNetwork(n_neurons)
        
        # Set up TSP-specific weights
        weights = np.zeros((n_neurons, n_neurons))
        
        # Penalty parameters
        A, B, C, D = 500, 500, 200, 100
        
        for i in range(n_cities):
            for j in range(n_cities):
                for k in range(n_cities):
                    for l in range(n_cities):
                        neuron_ij = i * n_cities + j
                        neuron_kl = k * n_cities + l
                        
                        # Row constraint
                        if i == k and j != l:
                            weights[neuron_ij, neuron_kl] -= A
                            
                        # Column constraint
                        if j == l and i != k:
                            weights[neuron_ij, neuron_kl] -= B
                            
                        # Global constraint
                        if i != k and j != l:
                            weights[neuron_ij, neuron_kl] -= C
                            
                        # Distance term
                        if k == (i + 1) % n_cities:
                            weights[neuron_ij, neuron_kl] -= D * distance_matrix[j, l]
                            
        self.network.weights = weights
        
        # Run network to find solution
        self.network.update_async(max_iterations=1000)
        
        # Extract tour from network state
        tour = []
        state_matrix = self.network.states.reshape(n_cities, n_cities)
        
        for step in range(n_cities):
            city = np.argmax(state_matrix[:, step])
            tour.append(city)
            
        return tour
    
    def solve_graph_coloring(self, adjacency_matrix: np.ndarray, n_colors: int) -> List[int]:
        """Solve graph coloring problem."""
        n_vertices = len(adjacency_matrix)
        n_neurons = n_vertices * n_colors
        self.network = HopfieldNetwork(n_neurons)
        
        # Set up weights for graph coloring
        weights = np.zeros((n_neurons, n_neurons))
        
        # Penalty for adjacent vertices having same color
        for i in range(n_vertices):
            for j in range(n_vertices):
                if adjacency_matrix[i, j] == 1:
                    for c in range(n_colors):
                        neuron_ic = i * n_colors + c
                        neuron_jc = j * n_colors + c
                        weights[neuron_ic, neuron_jc] = -1000
                        
        # Penalty for vertex having multiple colors
        for i in range(n_vertices):
            for c1 in range(n_colors):
                for c2 in range(n_colors):
                    if c1 != c2:
                        neuron_ic1 = i * n_colors + c1
                        neuron_ic2 = i * n_colors + c2
                        weights[neuron_ic1, neuron_ic2] = -1000
                        
        self.network.weights = weights
        
        # Initialize with random valid coloring
        initial_state = np.zeros(n_neurons)
        for i in range(n_vertices):
            color = np.random.randint(n_colors)
            initial_state[i * n_colors + color] = 1
        self.network.states = np.where(initial_state > 0, 1, -1)
        
        # Run network
        self.network.update_async(max_iterations=500)
        
        # Extract coloring
        coloring = []
        state_matrix = self.network.states.reshape(n_vertices, n_colors)
        
        for i in range(n_vertices):
            color = np.argmax(state_matrix[i])
            coloring.append(color)
            
        return coloring


class ErrorCorrectionSystem:
    """Use Hopfield Networks for error correction in data transmission."""
    
    def __init__(self, codeword_length: int):
        self.codeword_length = codeword_length
        self.network = HopfieldNetwork(codeword_length)
        self.valid_codewords = []
        
    def generate_hamming_codewords(self, data_bits: int) -> List[np.ndarray]:
        """Generate Hamming codewords for error correction."""
        # Calculate number of parity bits needed
        parity_bits = 0
        while (2 ** parity_bits) < (data_bits + parity_bits + 1):
            parity_bits += 1
            
        total_bits = data_bits + parity_bits
        codewords = []
        
        # Generate all possible data combinations
        for i in range(2 ** data_bits):
            data = [(i >> j) & 1 for j in range(data_bits)]
            codeword = self._create_hamming_code(data, parity_bits)
            # Convert to Hopfield format (-1, 1)
            codeword = np.array([1 if bit else -1 for bit in codeword])
            codewords.append(codeword)
            
        return codewords
    
    def _create_hamming_code(self, data: List[int], parity_bits: int) -> List[int]:
        """Create Hamming code from data bits."""
        n = len(data) + parity_bits
        code = [0] * n
        
        # Place data bits
        j = 0
        for i in range(n):
            if (i + 1) & i == 0:  # Skip positions that are powers of 2
                continue
            code[i] = data[j]
            j += 1
            
        # Calculate parity bits
        for i in range(parity_bits):
            parity_pos = (1 << i) - 1
            parity = 0
            for j in range(parity_pos, n, 2 * (parity_pos + 1)):
                for k in range(min(parity_pos + 1, n - j)):
                    if j + k != parity_pos:
                        parity ^= code[j + k]
            code[parity_pos] = parity
            
        return code
    
    def train_error_corrector(self, codewords: List[np.ndarray]):
        """Train network to recognize valid codewords."""
        self.valid_codewords = codewords
        patterns = np.array(codewords)
        self.network.train(patterns)
        
    def correct_errors(self, received_data: np.ndarray) -> Dict:
        """Correct errors in received data."""
        # Use Hopfield network to recall nearest valid codeword
        corrected = self.network.recall(received_data, max_iterations=50)
        
        # Find which valid codeword it matches
        best_match = None
        best_similarity = -1
        
        for codeword in self.valid_codewords:
            similarity = np.mean(corrected == codeword)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = codeword
                
        # Count number of corrected bits
        n_errors = np.sum(received_data != best_match) // 2
        
        return {
            'corrected_data': best_match,
            'n_errors_corrected': n_errors,
            'confidence': best_similarity,
            'success': best_similarity > 0.9
        }


class PatternCompletion:
    """Use Hopfield Networks for pattern completion and prediction."""
    
    def __init__(self, pattern_size: int):
        self.pattern_size = pattern_size
        self.network = HopfieldNetwork(pattern_size)
        self.pattern_library = {}
        
    def add_pattern_sequence(self, name: str, patterns: List[np.ndarray]):
        """Add a sequence of patterns to the library."""
        self.pattern_library[name] = patterns
        
    def train_on_sequences(self):
        """Train network on all pattern sequences."""
        all_patterns = []
        for patterns in self.pattern_library.values():
            all_patterns.extend(patterns)
            
        if all_patterns:
            patterns_array = np.array(all_patterns)
            self.network.train(patterns_array)
            
    def complete_partial_pattern(self, partial_pattern: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """Complete a partial pattern where mask indicates known values."""
        # Initialize with partial pattern
        current = partial_pattern.copy()
        
        # Set unknown values to random
        unknown_indices = np.where(mask == 0)[0]
        current[unknown_indices] = np.random.choice([-1, 1], size=len(unknown_indices))
        
        # Use network to complete
        completed = self.network.recall(current)
        
        # Ensure known values are preserved
        completed[mask == 1] = partial_pattern[mask == 1]
        
        return completed
    
    def predict_next_in_sequence(self, sequence_start: List[np.ndarray]) -> np.ndarray:
        """Predict the next pattern in a sequence."""
        # Create a combined pattern from sequence
        combined = np.concatenate(sequence_start[-2:])  # Use last 2 patterns
        
        # Add noise to create variation
        noisy = self.network.add_noise(combined, 0.1)
        
        # Recall and extract predicted next pattern
        recalled = self.network.recall(noisy[:self.pattern_size])
        
        return recalled