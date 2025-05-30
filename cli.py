#!/usr/bin/env python3
import argparse
import numpy as np
from hopfield_network import HopfieldNetwork, create_example_network
import json


def print_network_state(network: HopfieldNetwork, iteration: int = None):
    """Pretty print the network state."""
    if iteration is not None:
        print(f"\nIteration {iteration}:")
    print(f"States: {' '.join(['+' if s > 0 else '-' for s in network.states])}")
    print(f"Energy: {network.energy():.2f}")


def interactive_mode(network: HopfieldNetwork):
    """Interactive mode for manual network manipulation."""
    print("\nInteractive Mode Commands:")
    print("  flip <n>    - Flip neuron n")
    print("  update      - Run one update iteration")
    print("  solve       - Run until convergence")
    print("  energy      - Show current energy")
    print("  states      - Show current states")
    print("  weights     - Show weight matrix")
    print("  quit        - Exit")
    
    while True:
        try:
            cmd = input("\n> ").strip().lower().split()
            if not cmd:
                continue
                
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'flip' and len(cmd) > 1:
                neuron = int(cmd[1])
                if 0 <= neuron < network.n_neurons:
                    network.states[neuron] *= -1
                    print(f"Flipped neuron {neuron}")
                    print_network_state(network)
                else:
                    print(f"Invalid neuron index: {neuron}")
            elif cmd[0] == 'update':
                changed = network.update_sync()
                print_network_state(network)
                if not changed:
                    print("Network is stable!")
            elif cmd[0] == 'solve':
                iterations = network.update_async()
                print(f"Converged in {iterations} iterations")
                print_network_state(network)
            elif cmd[0] == 'energy':
                print(f"Current energy: {network.energy():.2f}")
            elif cmd[0] == 'states':
                print_network_state(network)
            elif cmd[0] == 'weights':
                print("\nWeight matrix:")
                print(network.weights)
            else:
                print("Unknown command")
        except (ValueError, IndexError):
            print("Invalid command format")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Hopfield Network CLI - Fast implementation with various modes"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Example command
    example_parser = subparsers.add_parser('example', 
        help='Run the 4-neuron example from the Java code')
    example_parser.add_argument('--verbose', '-v', action='store_true',
        help='Show each iteration')
    
    # Custom network command
    custom_parser = subparsers.add_parser('custom',
        help='Create custom network with manual connections')
    custom_parser.add_argument('neurons', type=int, 
        help='Number of neurons')
    custom_parser.add_argument('--connections', '-c', type=str,
        help='JSON file with connections [(i,j,weight), ...]')
    custom_parser.add_argument('--interactive', '-i', action='store_true',
        help='Enter interactive mode')
    
    # Pattern recognition command
    pattern_parser = subparsers.add_parser('patterns',
        help='Pattern storage and recall demo')
    pattern_parser.add_argument('--size', '-s', type=int, default=25,
        help='Pattern size (default: 25)')
    pattern_parser.add_argument('--num-patterns', '-n', type=int, default=3,
        help='Number of patterns to store (default: 3)')
    pattern_parser.add_argument('--noise', type=float, default=0.2,
        help='Noise level for recall test (default: 0.2)')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark',
        help='Performance benchmark')
    bench_parser.add_argument('--sizes', nargs='+', type=int,
        default=[10, 50, 100, 500, 1000],
        help='Network sizes to test')
    
    args = parser.parse_args()
    
    if args.command == 'example':
        print("Running 4-neuron example (Java equivalent)...")
        network = create_example_network()
        
        print("\nInitial state:")
        print_network_state(network)
        
        if args.verbose:
            for i in range(20):
                if not network.update_sync():
                    print(f"\nConverged at iteration {i}")
                    break
                print_network_state(network, i+1)
        else:
            iterations = network.update_async()
            print(f"\nConverged after {iterations} iterations")
            
        print("\nFinal state:")
        print_network_state(network)
        
    elif args.command == 'custom':
        network = HopfieldNetwork(args.neurons)
        
        if args.connections:
            with open(args.connections, 'r') as f:
                connections = json.load(f)
                network.set_weights(connections)
                print(f"Loaded {len(connections)} connections")
        
        if args.interactive:
            interactive_mode(network)
        else:
            print_network_state(network)
            iterations = network.update_async()
            print(f"\nConverged after {iterations} iterations")
            print_network_state(network)
            
    elif args.command == 'patterns':
        import time
        
        print(f"Creating network with {args.size} neurons...")
        network = HopfieldNetwork(args.size)
        
        # Generate random patterns
        patterns = np.random.choice([-1, 1], 
            size=(args.num_patterns, args.size))
        
        print(f"Training on {args.num_patterns} patterns...")
        start = time.time()
        network.train(patterns)
        train_time = time.time() - start
        print(f"Training completed in {train_time:.3f}s")
        
        # Test recall
        print(f"\nTesting recall with {args.noise:.0%} noise...")
        successes = 0
        
        for i, pattern in enumerate(patterns):
            noisy = network.add_noise(pattern, args.noise)
            start = time.time()
            recalled = network.recall(noisy)
            recall_time = time.time() - start
            
            accuracy = np.mean(recalled == pattern)
            success = np.array_equal(recalled, pattern)
            successes += success
            
            print(f"Pattern {i+1}: {accuracy:.1%} accuracy, "
                  f"{'✓' if success else '✗'} perfect recall "
                  f"({recall_time:.3f}s)")
        
        print(f"\nOverall: {successes}/{args.num_patterns} perfect recalls")
        
    elif args.command == 'benchmark':
        import time
        
        print("Performance Benchmark")
        print("-" * 50)
        print(f"{'Size':>10} {'Train(ms)':>12} {'Recall(ms)':>12} {'Energy(μs)':>12}")
        print("-" * 50)
        
        for size in args.sizes:
            network = HopfieldNetwork(size)
            patterns = np.random.choice([-1, 1], size=(3, size))
            
            # Benchmark training
            start = time.time()
            network.train(patterns)
            train_time = (time.time() - start) * 1000
            
            # Benchmark recall
            noisy = network.add_noise(patterns[0], 0.2)
            start = time.time()
            network.recall(noisy, max_iterations=50)
            recall_time = (time.time() - start) * 1000
            
            # Benchmark energy calculation
            times = []
            for _ in range(1000):
                start = time.time()
                network.energy()
                times.append((time.time() - start) * 1e6)
            energy_time = np.mean(times)
            
            print(f"{size:>10} {train_time:>12.2f} {recall_time:>12.2f} {energy_time:>12.2f}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()