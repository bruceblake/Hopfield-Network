# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run core functionality demo
python simple_demo.py

# Run web interface
streamlit run demo_app.py

# Run CLI commands
python hopfield_cli.py --help
python hopfield_cli.py create --size 100 --name TestNet --save
python hopfield_cli.py app password --partial-password "p***word"

# Run performance benchmark
python benchmark.py --core-only

# Run tests
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html

# Docker deployment
docker build -t hopfield-toolkit .
docker run --rm hopfield-toolkit
docker-compose up hopfield-demo
```

## Architecture Overview

This is a **professional-grade Hopfield Network toolkit** designed for real-world applications rather than educational demos. The implementation uses NumPy for high-performance vectorized operations.

### Core Components

1. **hopfield_toolkit.py**: Core network implementation
   - `HopfieldNetwork` class with NumPy-optimized operations
   - Multiple training methods: Hebbian learning and pseudoinverse rule
   - Advanced features: stability analysis, capacity testing, energy tracking
   - Professional APIs: save/load, detailed statistics, error handling

2. **applications.py**: Real-world application modules
   - `ImageRestoration`: Image denoising using patch-based Hopfield networks
   - `PasswordRecovery`: Password completion from partial information
   - `OptimizationSolver`: Traveling Salesman Problem solving
   - `PatternCompletion`: Text/sequence completion system

3. **demo_app.py**: Professional Streamlit web interface
   - Interactive network visualization with Plotly
   - Multiple application modes with real demonstrations
   - Real-time performance metrics and analysis
   - Pattern drawing interfaces and result visualization

4. **hopfield_cli.py**: Comprehensive command-line interface
   - Full network lifecycle: create, train, test, analyze
   - Application-specific commands for all use cases
   - Professional output formatting and error handling
   - JSON/pickle file format support

5. **benchmark.py**: Performance analysis suite
   - Comprehensive benchmarking across network sizes
   - Memory usage tracking and performance profiling
   - Scalability analysis and capacity testing
   - Automated plot generation and result saving

6. **tests/**: Comprehensive test suite
   - Core functionality tests with edge cases
   - Application integration tests
   - CLI command testing with temporary directories
   - Performance benchmarks with pytest-benchmark

### Key Implementation Details

- **State Representation**: Binary states {-1, 1} for mathematical convenience
- **Weight Matrix**: Symmetric with zero diagonal, stored as full matrix
- **Energy Function**: E = -0.5 Σᵢⱼ wᵢⱼ sᵢ sⱼ (standard Hopfield energy)
- **Training Methods**:
  - Hebbian: wᵢⱼ = (1/n) Σₖ pᵢᵏ pⱼᵏ over all patterns k
  - Pseudoinverse: wᵢⱼ = P @ pinv(P) for better capacity
- **Update Rules**: 
  - Asynchronous: Random sequential neuron updates (guaranteed convergence)
  - Synchronous: Parallel update of all neurons (faster but may cycle)
- **Performance**: Fully vectorized NumPy operations, O(n²) complexity per iteration
- **Capacity**: Theoretical ~0.138N patterns, practical ~0.05N patterns

### Real-World Applications

1. **Letter Recognition** (`demo_app.py` Letter Recognition mode)
   - 5x5 binary letter patterns with noise tolerance
   - Interactive pattern corruption and recognition testing
   - Real-time accuracy and convergence metrics

2. **Password Recovery** (`applications.py` PasswordRecovery class)
   - Character encoding with one-hot representation
   - Pattern completion from partial password information
   - Confidence scoring and multiple candidate generation

3. **Route Optimization** (`applications.py` OptimizationSolver class)
   - TSP formulation using constraint satisfaction
   - Energy minimization for combinatorial optimization
   - Solution validation and tour extraction

4. **Text Completion** (`applications.py` PatternCompletion class)
   - Sequence pattern learning with custom vocabularies
   - Context-aware completion from partial text
   - Multi-candidate generation with ranking

### Testing and Validation

- **Core Tests**: Network creation, training, recall, energy calculation
- **Application Tests**: End-to-end workflows for all applications
- **CLI Tests**: Command execution with temporary file handling
- **Performance Tests**: Benchmarking with pytest-benchmark integration
- **Error Handling**: Invalid inputs, capacity warnings, convergence failures
- **Docker Testing**: Containerized test execution for reproducibility

### Development Guidelines

- **Error Handling**: Professional exception handling with informative messages
- **Performance**: Always use vectorized NumPy operations, avoid Python loops
- **Memory**: Efficient memory management for large networks
- **CLI Design**: Follow Unix conventions, provide comprehensive help
- **Web Interface**: Responsive design with real-time feedback
- **Testing**: Maintain high test coverage, include performance benchmarks
- **Documentation**: Clear docstrings, type hints, usage examples

### Common Issues and Solutions

1. **Capacity Warnings**: Network automatically warns when pattern count exceeds theoretical limits
2. **Non-Convergence**: TSP solver may not find valid solutions (NP-hard problem)
3. **Memory Usage**: Large networks (>1000 neurons) require significant RAM
4. **Docker Dependencies**: Some tests require specific file paths for CLI testing
5. **Streamlit Errors**: Pattern grid sizing issues resolved in demo_app.py

### Performance Characteristics

- **Training**: 0.1-50ms depending on network size and pattern count
- **Recall**: 0.1-200ms depending on network size and iterations
- **Memory**: ~n² * 8 bytes for weight matrix (dominant factor)
- **Scalability**: Linear scaling up to ~2000 neurons, then memory-bound

This toolkit is designed for **researchers and practitioners** who need reliable, high-performance Hopfield Network implementations for real applications, not just academic demonstrations.