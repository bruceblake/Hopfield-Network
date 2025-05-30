# Hopfield Network Toolkit - Usage Guide

## Quick Start

### 1. Start the Web UI
```bash
docker-compose up hopfield-demo
# Open http://localhost:8501 in your browser
```

### 2. CLI Examples

```bash
# Create and train a network
docker-compose run --rm hopfield-toolkit python hopfield_cli.py create --size 100
docker-compose run --rm hopfield-toolkit python hopfield_cli.py train --patterns 5 --save

# Test pattern recall
docker-compose run --rm hopfield-toolkit python hopfield_cli.py test --noise 0.1

# Run applications
docker-compose run --rm hopfield-toolkit python hopfield_cli.py password --length 8
docker-compose run --rm hopfield-toolkit python hopfield_cli.py tsp --cities 5
docker-compose run --rm hopfield-toolkit python hopfield_cli.py completion --length 10
docker-compose run --rm hopfield-toolkit python hopfield_cli.py image --size 32

# Analyze network
docker-compose run --rm hopfield-toolkit python hopfield_cli.py analyze
```

### 3. Python API

```python
from hopfield_toolkit import HopfieldNetwork
import numpy as np

# Create network
network = HopfieldNetwork(n_neurons=100)

# Train with patterns
patterns = np.random.choice([-1, 1], size=(5, 100))
network.train_hebbian(patterns)

# Recall pattern with noise
noisy = network.add_noise(patterns[0], 0.2)
recalled = network.recall(noisy)
```

### 4. Run Tests
```bash
docker-compose run --rm hopfield-toolkit pytest -v
```

### 5. Run Benchmarks
```bash
docker-compose run --rm hopfield-benchmark
```

## Applications

### Password Recovery
- Stores passwords as patterns
- Can recover passwords with missing/corrupted characters
- Useful for password hint systems

### Image Restoration
- Removes noise from images
- Reconstructs missing parts
- Works with binary patterns

### TSP Optimization
- Solves traveling salesman problems
- Finds near-optimal routes
- Visualizes solutions

### Pattern Completion
- Completes partial sequences
- Works with text, numbers, or symbols
- Useful for predictive systems

## Tips

1. **Network Size**: Choose size based on pattern complexity
2. **Training**: More patterns = lower recall accuracy
3. **Noise Level**: Start with 10-20% for testing
4. **Max Iterations**: Usually converges within 10-20 iterations

## Troubleshooting

- **Port 8501 in use**: Change port in docker-compose.yml
- **Memory issues**: Reduce network size or pattern count
- **Slow performance**: Use pseudoinverse training for large networks