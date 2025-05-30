#!/usr/bin/env python3
"""
Professional Hopfield Network CLI Tool
Advanced features for research and practical applications
"""

import argparse
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import time

from hopfield_toolkit import HopfieldNetwork, HopfieldAnalyzer
from applications import ImageRestoration, PasswordRecovery, OptimizationSolver, PatternCompletion


class HopfieldCLI:
    """Professional CLI interface for Hopfield Networks."""
    
    def __init__(self):
        self.networks = {}
        self.current_network = None
        self.state_file = Path(".hopfield_state.pkl")
        self.load_state()
        
    def save_state(self):
        """Save CLI state to file."""
        try:
            state = {
                'current_network_name': self.current_network.name if self.current_network else None,
                'network_names': list(self.networks.keys())
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
        except:
            pass  # Ignore errors
    
    def load_state(self):
        """Load CLI state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                # We can't restore actual networks here, just the names
        except:
            pass  # Ignore errors
        
    def create_network(self, args):
        """Create a new Hopfield network."""
        network = HopfieldNetwork(args.size, args.name)
        self.networks[args.name] = network
        self.current_network = network
        
        print(f"Created network '{args.name}' with {args.size} neurons")
        
        # Auto-save network info
        if args.save:
            network.save(f"{args.name}.pkl")
            print(f"Saved to {args.name}.pkl")
    
    def load_network(self, args):
        """Load network from file."""
        try:
            network = HopfieldNetwork.load(args.file)
            self.networks[network.name] = network
            self.current_network = network
            print(f"Loaded network '{network.name}' from {args.file}")
            
            if hasattr(network, 'training_info') and network.training_info:
                print(f"Training info: {network.training_info}")
                
        except Exception as e:
            print(f"Error loading network: {e}")
            sys.exit(1)
    
    def train_network(self, args):
        """Train the current network."""
        # Try to find a network file if no current network
        if self.current_network is None:
            # Look for .pkl files in current directory
            pkl_files = list(Path.cwd().glob("*.pkl"))
            if pkl_files:
                # Load the most recent one
                try:
                    self.current_network = HopfieldNetwork.load(str(pkl_files[-1]))
                    self.networks[self.current_network.name] = self.current_network
                except:
                    pass
        
        if self.current_network is None:
            print("No network selected. Create or load a network first.")
            return
            
        # Load patterns
        if args.patterns_file:
            patterns = self.load_patterns(args.patterns_file)
        elif args.random_patterns:
            patterns = np.random.choice([-1, 1], size=(args.random_patterns, self.current_network.n_neurons))
        else:
            print("No training patterns specified")
            return
            
        print(f"Training on {len(patterns)} patterns...")
        
        # Choose training method
        if args.method == 'hebbian':
            stats = self.current_network.train_hebbian(patterns, normalize=args.normalize)
        elif args.method == 'pseudoinverse':
            stats = self.current_network.train_pseudoinverse(patterns)
        else:
            print(f"Unknown training method: {args.method}")
            return
            
        print("Training completed!")
        print(f"Capacity ratio: {stats.get('capacity_ratio', 'N/A'):.3f}")
        print(f"Training time: {stats.get('training_time', 0):.3f}s")
        
        if args.save:
            filename = f"{self.current_network.name}_trained.pkl"
            self.current_network.save(filename)
            print(f"Saved trained network to {filename}")
    
    def test_recall(self, args):
        """Test pattern recall."""
        # Try to load network if not available
        if self.current_network is None:
            pkl_files = list(Path.cwd().glob("*_trained.pkl"))
            if not pkl_files:
                pkl_files = list(Path.cwd().glob("*.pkl"))
            if pkl_files:
                try:
                    self.current_network = HopfieldNetwork.load(str(pkl_files[-1]))
                    self.networks[self.current_network.name] = self.current_network
                except:
                    pass
        
        if self.current_network is None or len(self.current_network.patterns) == 0:
            print("No trained network available")
            return
            
        results = []
        
        for i, pattern in enumerate(self.current_network.patterns):
            # Add noise
            noisy = self.current_network.add_noise(pattern, args.noise_level, args.noise_type)
            
            # Test recall
            start_time = time.time()
            result = self.current_network.recall(noisy, max_iterations=args.max_iterations, 
                                               async_update=args.async_update)
            recall_time = time.time() - start_time
            
            # Calculate accuracy
            accuracy = np.mean(result['final_states'] == pattern)
            perfect_recall = np.array_equal(result['final_states'], pattern)
            
            results.append({
                'pattern_id': i,
                'accuracy': accuracy,
                'perfect_recall': perfect_recall,
                'iterations': result['iterations'],
                'converged': result['converged'],
                'energy_reduction': result['energy_reduction'],
                'recall_time': recall_time
            })
            
            if args.verbose:
                print(f"Pattern {i}: {accuracy:.1%} accuracy, "
                      f"{'✓' if perfect_recall else '✗'} perfect, "
                      f"{result['iterations']} iterations ({recall_time:.3f}s)")
        
        # Summary statistics
        accuracies = [r['accuracy'] for r in results]
        perfect_recalls = sum(1 for r in results if r['perfect_recall'])
        avg_iterations = np.mean([r['iterations'] for r in results])
        avg_time = np.mean([r['recall_time'] for r in results])
        
        print(f"\nSummary:")
        print(f"Average accuracy: {np.mean(accuracies):.1%}")
        print(f"Perfect recalls: {perfect_recalls}/{len(results)}")
        print(f"Average iterations: {avg_iterations:.1f}")
        print(f"Average time: {avg_time:.3f}s")
        
        if args.save_results:
            with open(f"recall_results_{self.current_network.name}.json", 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to recall_results_{self.current_network.name}.json")
    
    def analyze_network(self, args):
        """Perform network analysis."""
        # Try to load network if not available
        if self.current_network is None:
            pkl_files = list(Path.cwd().glob("*.pkl"))
            if pkl_files:
                try:
                    self.current_network = HopfieldNetwork.load(str(pkl_files[-1]))
                    self.networks[self.current_network.name] = self.current_network
                except:
                    pass
        
        if self.current_network is None:
            print("No network selected")
            return
            
        print(f"Analyzing network: {self.current_network.name}")
        print("=" * 50)
        
        # Basic info
        print(f"Neurons: {self.current_network.n_neurons}")
        print(f"Stored patterns: {len(self.current_network.patterns)}")
        
        if self.current_network.training_info:
            info = self.current_network.training_info
            print(f"Training method: {info.get('method', 'hebbian')}")
            print(f"Capacity ratio: {info.get('capacity_ratio', 0):.3f}")
        
        # Weight analysis
        weights = self.current_network.weights
        print(f"\nWeight statistics:")
        print(f"  Mean: {np.mean(weights):.4f}")
        print(f"  Std: {np.std(weights):.4f}")
        print(f"  Range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  Sparsity: {np.mean(weights == 0):.1%}")
        
        # Stability analysis
        if self.current_network.patterns:
            stability = self.current_network.stability_analysis()
            stable_count = sum(1 for p in stability['pattern_stability'] if p['is_stable'])
            print(f"\nStability analysis:")
            print(f"  Stable patterns: {stable_count}/{len(self.current_network.patterns)}")
            
            if args.detailed:
                for i, info in enumerate(stability['pattern_stability']):
                    status = "✓" if info['is_stable'] else "✗"
                    print(f"  Pattern {i}: {status} (energy: {info['energy']:.2f}, "
                          f"violations: {info['violations']})")
        
        # Capacity test
        if args.capacity_test:
            print(f"\nRunning capacity test...")
            capacity_results = self.current_network.capacity_test()
            max_patterns = max(capacity_results['pattern_counts'])
            best_rate = max(capacity_results['success_rates'])
            print(f"  Tested up to {max_patterns} patterns")
            print(f"  Best success rate: {best_rate:.1%}")
            
            if args.save_plots:
                plt.figure(figsize=(10, 6))
                plt.plot(capacity_results['pattern_counts'], capacity_results['success_rates'], 'bo-')
                plt.xlabel('Number of Patterns')
                plt.ylabel('Success Rate')
                plt.title(f'Capacity Test - {self.current_network.name}')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'capacity_test_{self.current_network.name}.png', dpi=300, bbox_inches='tight')
                print(f"  Capacity plot saved to capacity_test_{self.current_network.name}.png")
        
        # Theoretical capacity
        theoretical = HopfieldAnalyzer.theoretical_capacity(self.current_network.n_neurons)
        print(f"\nTheoretical capacity limits:")
        print(f"  Hopfield: {theoretical['hopfield_capacity']} patterns")
        print(f"  Gardner: {theoretical['gardner_capacity']} patterns")
        print(f"  Practical: {theoretical['practical_capacity']} patterns")
    
    def application_commands(self, args):
        """Handle application-specific commands."""
        app_type = args.application
        
        if app_type == 'image':
            self.image_restoration(args)
        elif app_type == 'password':
            self.password_recovery(args)
        elif app_type == 'tsp':
            self.traveling_salesman(args)
        elif app_type == 'completion':
            self.pattern_completion(args)
        else:
            print(f"Unknown application: {app_type}")
    
    def image_restoration(self, args):
        """Image restoration application."""
        print("Image Restoration using Hopfield Networks")
        print("-" * 40)
        
        # Implementation would load actual images
        print("Note: This is a demo. In practice, load actual image files.")
        
        restorer = ImageRestoration(patch_size=args.patch_size)
        
        # Demo with synthetic data
        demo_images = [np.random.randint(0, 256, (64, 64)) for _ in range(3)]
        stats = restorer.train_on_images(demo_images, args.image_type)
        print(f"Trained on {len(demo_images)} images")
        print(f"Training time: {stats['training_time']:.3f}s")
        
        if args.test_image:
            print(f"Testing on {args.test_image}")
            # Would load and process actual image
    
    def password_recovery(self, args):
        """Password recovery application."""
        print("Password Recovery using Hopfield Networks")
        print("-" * 40)
        
        recovery = PasswordRecovery()
        recovery.setup_encoding(args.charset)
        
        if args.password_file:
            with open(args.password_file, 'r') as f:
                passwords = [line.strip() for line in f if line.strip()]
        else:
            # Demo passwords
            passwords = ["password123", "admin2024", "user1234", "secure99"]
            
        stats = recovery.train_on_passwords(passwords, args.max_length)
        print(f"Trained on {len(passwords)} passwords")
        
        if args.partial_password:
            known_pos = list(range(len(args.partial_password))) if args.known_positions is None else args.known_positions
            candidates = recovery.recover_password_advanced(args.partial_password, known_pos, args.max_length)
            
            print(f"\nRecovery results for '{args.partial_password}':")
            for i, (pwd, conf) in enumerate(candidates[:5]):
                print(f"  {i+1}. {pwd} (confidence: {conf:.3f})")
    
    def traveling_salesman(self, args):
        """TSP solver application."""
        print("Traveling Salesman Problem Solver")
        print("-" * 35)
        
        if args.cities_file:
            # Load cities from file
            with open(args.cities_file, 'r') as f:
                data = json.load(f)
                cities = np.array(data['cities'])
        else:
            # Random cities for demo
            np.random.seed(args.seed)
            cities = np.random.rand(args.n_cities, 2) * 100
            
        # Calculate distance matrix
        n_cities = len(cities)
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
        
        print(f"Solving TSP for {n_cities} cities...")
        
        solver = OptimizationSolver()
        result = solver.traveling_salesman(distance_matrix, args.max_iterations)
        
        if result['valid_solution']:
            print(f"✓ Valid solution found!")
            print(f"Tour: {result['tour']}")
            print(f"Total distance: {result['total_distance']:.2f}")
            print(f"Converged: {result['convergence_info']['converged']}")
            print(f"Iterations: {result['convergence_info']['iterations']}")
        else:
            print("✗ No valid solution found")
            print("Try increasing max_iterations or adjusting parameters")
    
    def pattern_completion(self, args):
        """Pattern completion application."""
        print("Pattern Completion using Hopfield Networks")
        print("-" * 42)
        
        completer = PatternCompletion(args.pattern_type)
        
        if args.sequences_file:
            with open(args.sequences_file, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            # Demo sequences
            sequences = ["HELLO WORLD", "PYTHON CODE", "NEURAL NETS", "HOPFIELD NET"]
            
        stats = completer.train_sequence_patterns(sequences)
        print(f"Trained on {len(sequences)} sequences")
        
        if args.partial_sequence:
            # Parse known positions
            known_pos = []
            for i, char in enumerate(args.partial_sequence):
                if char != '*':
                    known_pos.append(i)
                    
            completions = completer.complete_pattern(args.partial_sequence, known_pos)
            
            print(f"\nCompletions for '{args.partial_sequence}':")
            for i, completion in enumerate(completions):
                print(f"  {i+1}. {completion}")
    
    def load_patterns(self, filename: str) -> np.ndarray:
        """Load patterns from file."""
        ext = Path(filename).suffix.lower()
        
        if ext == '.json':
            with open(filename, 'r') as f:
                data = json.load(f)
                return np.array(data['patterns'])
        elif ext == '.npy':
            return np.load(filename)
        elif ext == '.txt':
            return np.loadtxt(filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Professional Hopfield Network Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hopfield_cli.py create --size 100 --name MyNet --save
  hopfield_cli.py load --file MyNet.pkl
  hopfield_cli.py train --random-patterns 5 --method hebbian --save
  hopfield_cli.py test --noise-level 0.2 --verbose
  hopfield_cli.py analyze --capacity-test --save-plots
  hopfield_cli.py app image --patch-size 8 --image-type document
  hopfield_cli.py app password --partial-password "p***word" --max-length 12
  hopfield_cli.py app tsp --n-cities 10 --max-iterations 1000
  hopfield_cli.py app completion --partial-sequence "H***O W****"
        """
    )
    
    cli = HopfieldCLI()
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create network
    create_parser = subparsers.add_parser('create', help='Create new network')
    create_parser.add_argument('--size', type=int, required=True, help='Number of neurons')
    create_parser.add_argument('--name', type=str, required=True, help='Network name')
    create_parser.add_argument('--save', action='store_true', help='Save after creation')
    create_parser.set_defaults(func=cli.create_network)
    
    # Load network
    load_parser = subparsers.add_parser('load', help='Load network from file')
    load_parser.add_argument('--file', type=str, required=True, help='Network file')
    load_parser.set_defaults(func=cli.load_network)
    
    # Train network
    train_parser = subparsers.add_parser('train', help='Train network')
    train_parser.add_argument('--patterns-file', type=str, help='Pattern file')
    train_parser.add_argument('--random-patterns', type=int, help='Generate N random patterns')
    train_parser.add_argument('--method', choices=['hebbian', 'pseudoinverse'], 
                             default='hebbian', help='Training method')
    train_parser.add_argument('--normalize', action='store_true', help='Normalize weights')
    train_parser.add_argument('--save', action='store_true', help='Save after training')
    train_parser.set_defaults(func=cli.train_network)
    
    # Test recall
    test_parser = subparsers.add_parser('test', help='Test pattern recall')
    test_parser.add_argument('--noise-level', type=float, default=0.2, help='Noise level')
    test_parser.add_argument('--noise-type', choices=['flip', 'gaussian', 'mask'], 
                            default='flip', help='Noise type')
    test_parser.add_argument('--max-iterations', type=int, default=100, help='Max iterations')
    test_parser.add_argument('--async-update', action='store_true', help='Use async updates')
    test_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    test_parser.add_argument('--save-results', action='store_true', help='Save results to file')
    test_parser.set_defaults(func=cli.test_recall)
    
    # Analyze network
    analyze_parser = subparsers.add_parser('analyze', help='Analyze network')
    analyze_parser.add_argument('--detailed', action='store_true', help='Detailed analysis')
    analyze_parser.add_argument('--capacity-test', action='store_true', help='Run capacity test')
    analyze_parser.add_argument('--save-plots', action='store_true', help='Save plots')
    analyze_parser.set_defaults(func=cli.analyze_network)
    
    # Applications
    app_parser = subparsers.add_parser('app', help='Run applications')
    app_subparsers = app_parser.add_subparsers(dest='application', help='Application type')
    
    # Image restoration
    img_parser = app_subparsers.add_parser('image', help='Image restoration')
    img_parser.add_argument('--patch-size', type=int, default=8, help='Patch size')
    img_parser.add_argument('--image-type', type=str, default='general', help='Image type')
    img_parser.add_argument('--test-image', type=str, help='Test image file')
    
    # Password recovery
    pwd_parser = app_subparsers.add_parser('password', help='Password recovery')
    pwd_parser.add_argument('--password-file', type=str, help='Password list file')
    pwd_parser.add_argument('--partial-password', type=str, help='Partial password (use * for unknown)')
    pwd_parser.add_argument('--known-positions', type=int, nargs='*', help='Known character positions')
    pwd_parser.add_argument('--max-length', type=int, default=12, help='Max password length')
    pwd_parser.add_argument('--charset', type=str, 
                           default='abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*',
                           help='Character set')
    
    # TSP solver
    tsp_parser = app_subparsers.add_parser('tsp', help='Traveling Salesman Problem')
    tsp_parser.add_argument('--cities-file', type=str, help='Cities JSON file')
    tsp_parser.add_argument('--n-cities', type=int, default=5, help='Number of random cities')
    tsp_parser.add_argument('--max-iterations', type=int, default=1000, help='Max iterations')
    tsp_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Pattern completion
    comp_parser = app_subparsers.add_parser('completion', help='Pattern completion')
    comp_parser.add_argument('--sequences-file', type=str, help='Sequences file')
    comp_parser.add_argument('--partial-sequence', type=str, help='Partial sequence (use * for unknown)')
    comp_parser.add_argument('--pattern-type', type=str, default='text', help='Pattern type')
    
    app_parser.set_defaults(func=cli.application_commands)
    
    # Parse and execute
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()