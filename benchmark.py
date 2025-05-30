#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for Hopfield Network Toolkit
"""

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import gc

from hopfield_toolkit import HopfieldNetwork, HopfieldAnalyzer
from applications import ImageRestoration, PasswordRecovery, OptimizationSolver, PatternCompletion


class HopfieldBenchmark:
    """Comprehensive benchmarking suite."""
    
    def __init__(self, save_results: bool = True, output_dir: str = "benchmark_results"):
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'system_info': self.get_system_info(),
            'core_benchmarks': {},
            'application_benchmarks': {},
            'scalability_tests': {},
            'memory_usage': {}
        }
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': psutil.sys.version,
            'numpy_version': np.__version__
        }
    
    def time_function(self, func, *args, **kwargs) -> Tuple[any, float, Dict]:
        """Time a function execution with memory monitoring."""
        # Force garbage collection
        gc.collect()
        
        # Get initial memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get final memory
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        execution_time = end_time - start_time
        memory_stats = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_peak_mb': memory_after,  # Approximate
            'memory_used_mb': memory_after - memory_before
        }
        
        return result, execution_time, memory_stats
    
    def benchmark_core_operations(self):
        """Benchmark core Hopfield Network operations."""
        print("Benchmarking core operations...")
        
        sizes = [50, 100, 200, 500, 1000]
        pattern_counts = [3, 5, 10]
        
        core_results = {
            'network_creation': {},
            'hebbian_training': {},
            'pseudoinverse_training': {},
            'pattern_recall': {},
            'energy_calculation': {}
        }
        
        for size in sizes:
            print(f"  Testing size {size}...")
            
            # Network creation
            _, time_create, mem_create = self.time_function(
                HopfieldNetwork, size, f"Benchmark_{size}"
            )
            core_results['network_creation'][size] = {
                'time': time_create,
                'memory': mem_create
            }
            
            for n_patterns in pattern_counts:
                if n_patterns > 0.2 * size:  # Skip if too many patterns
                    continue
                    
                patterns = np.random.choice([-1, 1], size=(n_patterns, size))
                
                # Hebbian training
                network = HopfieldNetwork(size, f"Hebbian_{size}_{n_patterns}")
                _, time_hebbian, mem_hebbian = self.time_function(
                    network.train_hebbian, patterns
                )
                
                key = f"{size}_{n_patterns}"
                core_results['hebbian_training'][key] = {
                    'time': time_hebbian,
                    'memory': mem_hebbian,
                    'size': size,
                    'patterns': n_patterns
                }
                
                # Pseudoinverse training (smaller sizes only)
                if size <= 200:
                    network_pseudo = HopfieldNetwork(size, f"Pseudo_{size}_{n_patterns}")
                    _, time_pseudo, mem_pseudo = self.time_function(
                        network_pseudo.train_pseudoinverse, patterns
                    )
                    core_results['pseudoinverse_training'][key] = {
                        'time': time_pseudo,
                        'memory': mem_pseudo,
                        'size': size,
                        'patterns': n_patterns
                    }
                
                # Pattern recall
                noisy_pattern = network.add_noise(patterns[0], 0.2)
                _, time_recall, mem_recall = self.time_function(
                    network.recall, noisy_pattern, max_iterations=100
                )
                core_results['pattern_recall'][key] = {
                    'time': time_recall,
                    'memory': mem_recall,
                    'size': size,
                    'patterns': n_patterns
                }
                
                # Energy calculation (multiple runs for accuracy)
                energy_times = []
                for _ in range(100):
                    start = time.perf_counter()
                    network.energy()
                    energy_times.append(time.perf_counter() - start)
                
                core_results['energy_calculation'][key] = {
                    'time_mean': np.mean(energy_times),
                    'time_std': np.std(energy_times),
                    'time_min': np.min(energy_times),
                    'size': size,
                    'patterns': n_patterns
                }
        
        self.results['core_benchmarks'] = core_results
        print("Core benchmarks completed!")
    
    def benchmark_applications(self):
        """Benchmark application-specific operations."""
        print("Benchmarking applications...")
        
        app_results = {}
        
        # Image Restoration
        print("  Image restoration...")
        restorer = ImageRestoration(patch_size=4)
        test_images = [np.random.randint(0, 256, (32, 32)) for _ in range(5)]
        
        _, time_img_train, mem_img_train = self.time_function(
            restorer.train_on_images, test_images[:3], "benchmark"
        )
        
        test_image = test_images[3]
        _, time_img_restore, mem_img_restore = self.time_function(
            restorer.restore_image, test_image, "benchmark"
        )
        
        app_results['image_restoration'] = {
            'training': {'time': time_img_train, 'memory': mem_img_train},
            'restoration': {'time': time_img_restore, 'memory': mem_img_restore}
        }
        
        # Password Recovery
        print("  Password recovery...")
        pwd_recovery = PasswordRecovery()
        pwd_recovery.setup_encoding()
        
        test_passwords = [f"password{i:03d}" for i in range(20)]
        _, time_pwd_train, mem_pwd_train = self.time_function(
            pwd_recovery.train_on_passwords, test_passwords, 12
        )
        
        _, time_pwd_recover, mem_pwd_recover = self.time_function(
            pwd_recovery.recover_password, "p***word123", [0, 4, 5, 6, 7], 12
        )
        
        app_results['password_recovery'] = {
            'training': {'time': time_pwd_train, 'memory': mem_pwd_train},
            'recovery': {'time': time_pwd_recover, 'memory': mem_pwd_recover}
        }
        
        # TSP Solver
        print("  TSP solver...")
        tsp_solver = OptimizationSolver()
        
        for n_cities in [4, 5, 6]:
            np.random.seed(42)
            cities = np.random.rand(n_cities, 2) * 100
            distance_matrix = np.zeros((n_cities, n_cities))
            
            for i in range(n_cities):
                for j in range(n_cities):
                    if i != j:
                        distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
            
            _, time_tsp, mem_tsp = self.time_function(
                tsp_solver.traveling_salesman, distance_matrix, 500
            )
            
            app_results[f'tsp_{n_cities}_cities'] = {
                'time': time_tsp,
                'memory': mem_tsp,
                'n_cities': n_cities
            }
        
        # Pattern Completion
        print("  Pattern completion...")
        pattern_completer = PatternCompletion()
        test_sequences = [f"SEQUENCE{i:02d}PATTERN" for i in range(10)]
        
        _, time_seq_train, mem_seq_train = self.time_function(
            pattern_completer.train_sequence_patterns, test_sequences
        )
        
        _, time_seq_complete, mem_seq_complete = self.time_function(
            pattern_completer.complete_pattern, "SEQ****", [0, 1, 2]
        )
        
        app_results['pattern_completion'] = {
            'training': {'time': time_seq_train, 'memory': mem_seq_train},
            'completion': {'time': time_seq_complete, 'memory': mem_seq_complete}
        }
        
        self.results['application_benchmarks'] = app_results
        print("Application benchmarks completed!")
    
    def benchmark_scalability(self):
        """Test scalability with increasing problem sizes."""
        print("Benchmarking scalability...")
        
        scalability_results = {}
        
        # Network size scaling
        sizes = [10, 50, 100, 250, 500, 1000, 2000]
        size_results = []
        
        for size in sizes:
            print(f"  Testing scalability for size {size}...")
            
            # Create network and patterns
            network = HopfieldNetwork(size, f"Scale_{size}")
            n_patterns = min(5, int(0.1 * size))  # Conservative pattern count
            patterns = np.random.choice([-1, 1], size=(n_patterns, size))
            
            # Time training
            _, train_time, train_mem = self.time_function(
                network.train_hebbian, patterns
            )
            
            # Time recall
            noisy = network.add_noise(patterns[0], 0.2)
            recall_result, recall_time, recall_mem = self.time_function(
                network.recall, noisy, max_iterations=50
            )
            
            # Time energy calculation
            energy_times = []
            for _ in range(10):
                start = time.perf_counter()
                network.energy()
                energy_times.append(time.perf_counter() - start)
            
            size_results.append({
                'size': size,
                'patterns': n_patterns,
                'train_time': train_time,
                'train_memory': train_mem['memory_used_mb'],
                'recall_time': recall_time,
                'recall_memory': recall_mem['memory_used_mb'],
                'recall_converged': recall_result['converged'],
                'recall_iterations': recall_result['iterations'],
                'energy_time_mean': np.mean(energy_times),
                'energy_time_std': np.std(energy_times)
            })
        
        scalability_results['size_scaling'] = size_results
        
        # Pattern count scaling
        network_size = 200
        pattern_counts = [1, 2, 5, 10, 15, 20, 25]
        pattern_results = []
        
        for n_patterns in pattern_counts:
            if n_patterns > 0.15 * network_size:  # Skip if too many
                continue
                
            print(f"  Testing {n_patterns} patterns...")
            
            network = HopfieldNetwork(network_size, f"PatternScale_{n_patterns}")
            patterns = np.random.choice([-1, 1], size=(n_patterns, network_size))
            
            # Time training
            _, train_time, train_mem = self.time_function(
                network.train_hebbian, patterns
            )
            
            # Test recall accuracy
            correct_recalls = 0
            total_recall_time = 0
            
            for pattern in patterns:
                noisy = network.add_noise(pattern, 0.2)
                recall_result, recall_time, _ = self.time_function(
                    network.recall, noisy, max_iterations=50
                )
                
                total_recall_time += recall_time
                if np.array_equal(recall_result['final_states'], pattern):
                    correct_recalls += 1
            
            pattern_results.append({
                'patterns': n_patterns,
                'train_time': train_time,
                'train_memory': train_mem['memory_used_mb'],
                'avg_recall_time': total_recall_time / n_patterns,
                'recall_accuracy': correct_recalls / n_patterns,
                'capacity_ratio': n_patterns / (0.138 * network_size)
            })
        
        scalability_results['pattern_scaling'] = pattern_results
        self.results['scalability_tests'] = scalability_results
        print("Scalability benchmarks completed!")
    
    def generate_plots(self):
        """Generate visualization plots."""
        print("Generating plots...")
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Core operations scaling
        if 'scalability_tests' in self.results and 'size_scaling' in self.results['scalability_tests']:
            size_data = self.results['scalability_tests']['size_scaling']
            sizes = [d['size'] for d in size_data]
            train_times = [d['train_time'] for d in size_data]
            recall_times = [d['recall_time'] for d in size_data]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training time scaling
            ax1.loglog(sizes, train_times, 'bo-', label='Training Time')
            ax1.set_xlabel('Network Size')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Training Time Scaling')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Recall time scaling
            ax2.loglog(sizes, recall_times, 'ro-', label='Recall Time')
            ax2.set_xlabel('Network Size')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Recall Time Scaling')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            if self.save_results:
                plt.savefig(self.output_dir / 'scaling_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Memory usage
        if 'scalability_tests' in self.results and 'size_scaling' in self.results['scalability_tests']:
            size_data = self.results['scalability_tests']['size_scaling']
            sizes = [d['size'] for d in size_data]
            memory_usage = [d['train_memory'] for d in size_data]
            
            plt.figure(figsize=(10, 6))
            plt.loglog(sizes, memory_usage, 'go-', label='Memory Usage')
            plt.xlabel('Network Size')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Scaling')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if self.save_results:
                plt.savefig(self.output_dir / 'memory_scaling.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Capacity vs accuracy
        if 'scalability_tests' in self.results and 'pattern_scaling' in self.results['scalability_tests']:
            pattern_data = self.results['scalability_tests']['pattern_scaling']
            capacity_ratios = [d['capacity_ratio'] for d in pattern_data]
            accuracies = [d['recall_accuracy'] for d in pattern_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(capacity_ratios, accuracies, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Capacity Ratio (patterns / theoretical limit)')
            plt.ylabel('Recall Accuracy')
            plt.title('Network Capacity vs Recall Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 1.1)
            
            if self.save_results:
                plt.savefig(self.output_dir / 'capacity_accuracy.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Plots generated!")
    
    def save_results_to_file(self):
        """Save results to JSON file."""
        if self.save_results:
            output_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {key: recursive_convert(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            converted_results = recursive_convert(self.results)
            
            with open(output_file, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # System info
        sys_info = self.results['system_info']
        print(f"System: {sys_info['cpu_count']} CPUs, {sys_info['memory_gb']:.1f}GB RAM")
        print(f"NumPy version: {sys_info['numpy_version']}")
        
        # Core performance highlights
        if 'scalability_tests' in self.results:
            size_data = self.results['scalability_tests'].get('size_scaling', [])
            if size_data:
                largest_test = max(size_data, key=lambda x: x['size'])
                print(f"\nLargest network tested: {largest_test['size']} neurons")
                print(f"Training time: {largest_test['train_time']:.3f}s")
                print(f"Memory usage: {largest_test['train_memory']:.1f}MB")
                print(f"Recall time: {largest_test['recall_time']:.3f}s")
        
        # Application performance
        if 'application_benchmarks' in self.results:
            app_bench = self.results['application_benchmarks']
            print(f"\nApplication Performance:")
            
            if 'image_restoration' in app_bench:
                img_time = app_bench['image_restoration']['restoration']['time']
                print(f"  Image restoration: {img_time:.3f}s")
            
            if 'password_recovery' in app_bench:
                pwd_time = app_bench['password_recovery']['recovery']['time']
                print(f"  Password recovery: {pwd_time:.3f}s")
            
            if 'tsp_5_cities' in app_bench:
                tsp_time = app_bench['tsp_5_cities']['time']
                print(f"  TSP (5 cities): {tsp_time:.3f}s")
        
        print("="*60)
    
    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("Starting comprehensive Hopfield Network benchmarks...")
        print("="*60)
        
        start_time = time.time()
        
        try:
            self.benchmark_core_operations()
            self.benchmark_applications()
            self.benchmark_scalability()
            self.generate_plots()
            
            total_time = time.time() - start_time
            print(f"\nAll benchmarks completed in {total_time:.1f} seconds")
            
            self.print_summary()
            self.save_results_to_file()
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            raise


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Hopfield Network Benchmark Suite")
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to files')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--core-only', action='store_true',
                       help='Run only core benchmarks')
    parser.add_argument('--apps-only', action='store_true',
                       help='Run only application benchmarks')
    parser.add_argument('--scalability-only', action='store_true',
                       help='Run only scalability tests')
    
    args = parser.parse_args()
    
    benchmark = HopfieldBenchmark(
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    if args.core_only:
        benchmark.benchmark_core_operations()
    elif args.apps_only:
        benchmark.benchmark_applications()
    elif args.scalability_only:
        benchmark.benchmark_scalability()
    else:
        benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()