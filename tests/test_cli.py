"""
Tests for CLI functionality
"""

import pytest
import subprocess
import tempfile
import json
import numpy as np
from pathlib import Path
import os
import sys

# Get the directory containing the CLI script
CLI_DIR = Path(__file__).parent.parent.absolute()
CLI_SCRIPT = CLI_DIR / "hopfield_cli.py"


def run_cli(*args, cwd=None, **kwargs):
    """Helper to run CLI commands with correct path."""
    cmd = [sys.executable, str(CLI_SCRIPT)] + list(args)
    # Set PYTHONPATH to include the project directory
    env = os.environ.copy()
    env['PYTHONPATH'] = str(CLI_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, **kwargs)


class TestCLICommands:
    """Test CLI command functionality."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "Professional Hopfield Network Toolkit" in result.stdout
    
    def test_create_network_command(self):
        """Test network creation via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cli("create", "--size", "10", "--name", "TestCLI", "--save", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Created network 'TestCLI'" in result.stdout
            assert Path(tmpdir, "TestCLI.pkl").exists()
    
    def test_train_command_with_random_patterns(self):
        """Test training command with random patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First create network and save it
            run_cli("create", "--size", "20", "--name", "TrainTest", "--save", cwd=tmpdir)
            
            # Then train
            result = run_cli("train", "--random-patterns", "3", "--method", "hebbian", "--save", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Training completed" in result.stdout
            assert "Capacity ratio" in result.stdout
    
    def test_test_recall_command(self):
        """Test recall testing command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train network
            run_cli("create", "--size", "15", "--name", "RecallTest", "--save", cwd=tmpdir)
            run_cli("train", "--random-patterns", "2", "--method", "hebbian", "--save", cwd=tmpdir)
            
            # Test recall
            result = run_cli("test", "--noise-level", "0.1", "--max-iterations", "50", "--verbose", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Summary:" in result.stdout
            assert "Average accuracy:" in result.stdout
    
    def test_analyze_command(self):
        """Test network analysis command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train network
            run_cli("create", "--size", "10", "--name", "AnalyzeTest", "--save", cwd=tmpdir)
            run_cli("train", "--random-patterns", "2", "--save", cwd=tmpdir)
            
            # Analyze
            result = run_cli("analyze", "--detailed", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Analyzing network:" in result.stdout
            assert "Weight statistics:" in result.stdout
            assert "Theoretical capacity limits:" in result.stdout
    
    def test_save_load_workflow(self):
        """Test save/load workflow via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create, train, and save
            run_cli("create", "--size", "8", "--name", "SaveLoadTest", "--save", cwd=tmpdir)
            run_cli("train", "--random-patterns", "2", "--save", cwd=tmpdir)
            
            # Load in new session
            result = run_cli("load", "--file", "SaveLoadTest_trained.pkl", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Loaded network 'SaveLoadTest'" in result.stdout


class TestApplicationCommands:
    """Test application-specific CLI commands."""
    
    def test_password_app_command(self):
        """Test password recovery application command."""
        result = run_cli("app", "password", "--partial-password", "p***word", "--max-length", "10")
        
        assert result.returncode == 0
        assert "Password Recovery" in result.stdout
        assert "Recovery results" in result.stdout
    
    def test_tsp_app_command(self):
        """Test TSP application command."""
        result = run_cli("app", "tsp", "--n-cities", "4", "--max-iterations", "100", "--seed", "42")
        
        assert result.returncode == 0
        assert "Traveling Salesman Problem" in result.stdout
        assert "Solving TSP for 4 cities" in result.stdout
    
    def test_completion_app_command(self):
        """Test pattern completion application command."""
        result = run_cli("app", "completion", "--partial-sequence", "H***O W****", "--pattern-type", "text")
        
        assert result.returncode == 0
        assert "Pattern Completion" in result.stdout
        assert "Completions for" in result.stdout
    
    def test_image_app_command(self):
        """Test image restoration application command."""
        result = run_cli("app", "image", "--patch-size", "4", "--image-type", "test")
        
        assert result.returncode == 0
        assert "Image Restoration" in result.stdout
        assert "Trained on" in result.stdout


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_missing_network_error(self):
        """Test error when no network is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_cli("train", "--random-patterns", "3", cwd=tmpdir)
            
            assert result.returncode == 0  # Should handle gracefully
            assert "No network selected" in result.stdout
    
    def test_invalid_file_error(self):
        """Test error with invalid file."""
        result = run_cli("load", "--file", "nonexistent.pkl")
        
        assert result.returncode == 1
        assert "Error loading network" in result.stdout
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        result = run_cli("invalid_command")
        
        # argparse returns 2 for invalid arguments
        assert result.returncode == 2
        assert "invalid choice" in result.stderr


class TestCLIPatternFiles:
    """Test CLI with pattern files."""
    
    def test_json_pattern_file(self):
        """Test training with JSON pattern file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pattern file
            patterns = {
                "patterns": [
                    [1, -1, 1, -1, 1],
                    [-1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1]
                ]
            }
            
            pattern_file = Path(tmpdir) / "patterns.json"
            with open(pattern_file, 'w') as f:
                json.dump(patterns, f)
            
            # Create network
            run_cli("create", "--size", "5", "--name", "PatternFileTest", "--save", cwd=tmpdir)
            
            # Train with pattern file
            result = run_cli("train", "--patterns-file", "patterns.json", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Training on 3 patterns" in result.stdout
    
    def test_numpy_pattern_file(self):
        """Test training with NumPy pattern file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pattern file
            patterns = np.random.choice([-1, 1], size=(4, 6))
            pattern_file = Path(tmpdir) / "patterns.npy"
            np.save(pattern_file, patterns)
            
            # Create network
            run_cli("create", "--size", "6", "--name", "NumpyTest", "--save", cwd=tmpdir)
            
            # Train with pattern file
            result = run_cli("train", "--patterns-file", "patterns.npy", cwd=tmpdir)
            
            assert result.returncode == 0
            assert "Training on 4 patterns" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])