#!/usr/bin/env python3
"""
Professional Hopfield Network Demo Application
Fixed version with proper error handling and real use cases
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from io import BytesIO
import base64

from hopfield_toolkit import HopfieldNetwork, HopfieldAnalyzer
from applications import ImageRestoration, PasswordRecovery, OptimizationSolver, PatternCompletion


# Page configuration
st.set_page_config(
    page_title="Hopfield Network Professional Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class HopfieldDemo:
    """Professional demo application."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'networks' not in st.session_state:
            st.session_state.networks = {}
        if 'current_network' not in st.session_state:
            st.session_state.current_network = None
        if 'demo_data' not in st.session_state:
            st.session_state.demo_data = self.generate_demo_data()
    
    def generate_demo_data(self):
        """Generate demonstration data for applications."""
        np.random.seed(42)  # Reproducible results
        
        return {
            'letter_patterns': self.create_letter_patterns(),
            'password_database': self.create_password_database(),
            'tsp_cities': self.create_tsp_instance(),
            'text_sequences': self.create_text_sequences(),
            'noise_examples': self.create_noise_examples()
        }
    
    def create_letter_patterns(self):
        """Create recognizable letter patterns."""
        patterns = {}
        
        # 5x5 letter patterns
        patterns['A'] = np.array([
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1]
        ])
        
        patterns['E'] = np.array([
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1]
        ])
        
        patterns['I'] = np.array([
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [1, 1, 1, 1, 1]
        ])
        
        patterns['O'] = np.array([
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1]
        ])
        
        patterns['U'] = np.array([
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1]
        ])
        
        return patterns
    
    def create_password_database(self):
        """Create realistic password database."""
        return [
            "password123", "admin2024", "user1234", "secure99",
            "login2024", "access01", "system24", "manager1",
            "database2024", "network123", "server99", "backup01"
        ]
    
    def create_tsp_instance(self):
        """Create TSP city instance."""
        city_names = ["New York", "Boston", "Philadelphia", "Washington", "Atlanta"]
        np.random.seed(42)
        coordinates = np.random.rand(len(city_names), 2) * 100
        
        return {
            'names': city_names,
            'coordinates': coordinates,
            'distance_matrix': self.calculate_distance_matrix(coordinates)
        }
    
    def calculate_distance_matrix(self, coordinates):
        """Calculate distance matrix from coordinates."""
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        return distances
    
    def create_text_sequences(self):
        """Create text sequence patterns."""
        return [
            "ARTIFICIAL INTELLIGENCE",
            "MACHINE LEARNING MODELS",
            "NEURAL NETWORK SYSTEMS",
            "DEEP LEARNING METHODS",
            "HOPFIELD NETWORK DEMO",
            "PATTERN RECOGNITION AI"
        ]
    
    def create_noise_examples(self):
        """Create examples of different noise types."""
        base_pattern = np.array([1, 1, -1, -1, 1, -1, 1, 1, -1, -1])
        
        examples = {}
        for noise_level in [0.1, 0.2, 0.3, 0.4]:
            network = HopfieldNetwork(10, "NoiseDemo")
            examples[f"{noise_level:.0%}"] = {
                'flip': network.add_noise(base_pattern, noise_level, 'flip'),
                'gaussian': network.add_noise(base_pattern, noise_level, 'gaussian'),
                'mask': network.add_noise(base_pattern, noise_level, 'mask')
            }
        
        return {'base': base_pattern, 'noisy': examples}
    
    def render_header(self):
        """Render application header."""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🧠 Hopfield Network Professional Toolkit")
            st.markdown("*Advanced associative memory systems for real-world applications*")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.header("🔧 Control Panel")
            
            # Application selection
            app_mode = st.selectbox(
                "Select Application",
                [
                    "📊 Network Analysis Dashboard",
                    "🔤 Letter Recognition",
                    "🔐 Password Recovery",
                    "🗺️ Route Optimization (TSP)",
                    "📝 Text Completion",
                    "🎯 Custom Network Builder"
                ]
            )
            
            st.markdown("---")
            
            # Global settings
            st.subheader("⚙️ Global Settings")
            
            max_iterations = st.slider("Max Iterations", 10, 500, 100)
            noise_level = st.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)
            
            update_method = st.radio(
                "Update Method",
                ["Asynchronous", "Synchronous"],
                help="Asynchronous updates neurons randomly, Synchronous updates all at once"
            )
            
            st.markdown("---")
            
            # Performance info
            st.subheader("📈 Performance Info")
            if st.session_state.current_network:
                net = st.session_state.current_network
                st.metric("Network Size", f"{net.n_neurons} neurons")
                st.metric("Stored Patterns", len(net.patterns))
                
                if net.training_info:
                    capacity_ratio = net.training_info.get('capacity_ratio', 0)
                    st.metric("Capacity Usage", f"{capacity_ratio:.1%}")
            
            return app_mode, max_iterations, noise_level, update_method == "Asynchronous"
    
    def render_network_dashboard(self):
        """Render network analysis dashboard."""
        st.header("📊 Network Analysis Dashboard")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Network Creation")
            
            with st.form("create_network"):
                name = st.text_input("Network Name", "AnalysisNet")
                size = st.number_input("Number of Neurons", 10, 1000, 100, 10)
                n_patterns = st.number_input("Random Patterns", 1, 20, 5)
                
                submitted = st.form_submit_button("Create & Train Network")
                
                if submitted:
                    with st.spinner("Creating and training network..."):
                        # Create network
                        network = HopfieldNetwork(size, name)
                        
                        # Generate and train on random patterns
                        patterns = np.random.choice([-1, 1], size=(n_patterns, size))
                        training_stats = network.train_hebbian(patterns)
                        
                        # Store in session
                        st.session_state.networks[name] = network
                        st.session_state.current_network = network
                        
                        st.success(f"Created network '{name}' with {size} neurons")
                        st.json(training_stats)
        
        with col2:
            st.subheader("Network Selection")
            
            if st.session_state.networks:
                selected_name = st.selectbox(
                    "Select Network",
                    list(st.session_state.networks.keys())
                )
                
                if st.button("Load Selected Network"):
                    st.session_state.current_network = st.session_state.networks[selected_name]
                    st.success(f"Loaded network: {selected_name}")
            else:
                st.info("No networks created yet")
        
        # Analysis section
        if st.session_state.current_network:
            st.markdown("---")
            self.render_network_analysis()
    
    def render_network_analysis(self):
        """Render detailed network analysis."""
        network = st.session_state.current_network
        
        st.subheader(f"Analysis: {network.name}")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Neurons", network.n_neurons)
        with col2:
            st.metric("Patterns", len(network.patterns))
        with col3:
            capacity_ratio = len(network.patterns) / (0.138 * network.n_neurons)
            st.metric("Capacity Ratio", f"{capacity_ratio:.2f}")
        with col4:
            if network.training_info:
                train_time = network.training_info.get('training_time', 0)
                st.metric("Train Time", f"{train_time:.3f}s")
        
        # Weight matrix visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weight Matrix Heatmap")
            
            if network.n_neurons <= 50:  # Only for small networks
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(network.weights, cmap='RdBu_r', aspect='auto')
                ax.set_title(f"Weight Matrix ({network.n_neurons}×{network.n_neurons})")
                ax.set_xlabel("Neuron j")
                ax.set_ylabel("Neuron i")
                plt.colorbar(im, ax=ax, label="Weight")
                st.pyplot(fig)
            else:
                st.info("Weight matrix too large to display (>50 neurons)")
        
        with col2:
            st.subheader("Weight Statistics")
            
            weights = network.weights
            weight_stats = {
                "Mean": float(np.mean(weights)),
                "Std Dev": float(np.std(weights)),
                "Min": float(np.min(weights)),
                "Max": float(np.max(weights)),
                "Sparsity": float(np.mean(weights == 0))
            }
            
            for stat, value in weight_stats.items():
                if stat == "Sparsity":
                    st.metric(stat, f"{value:.1%}")
                else:
                    st.metric(stat, f"{value:.4f}")
        
        # Stability analysis
        if st.button("Run Stability Analysis"):
            with st.spinner("Analyzing stability..."):
                stability = network.stability_analysis()
                
                if 'pattern_stability' in stability:
                    st.subheader("Pattern Stability")
                    
                    stability_data = []
                    for i, info in enumerate(stability['pattern_stability']):
                        stability_data.append({
                            'Pattern': i,
                            'Stable': '✓' if info['is_stable'] else '✗',
                            'Energy': info['energy'],
                            'Violations': info['violations']
                        })
                    
                    df = pd.DataFrame(stability_data)
                    st.dataframe(df, use_container_width=True)
                    
                    stable_count = sum(1 for info in stability['pattern_stability'] if info['is_stable'])
                    st.metric("Stable Patterns", f"{stable_count}/{len(stability['pattern_stability'])}")
    
    def render_letter_recognition(self, max_iterations, noise_level, async_update):
        """Render letter recognition demo."""
        st.header("🔤 Letter Recognition System")
        
        st.markdown("""
        This demo shows how Hopfield Networks can recognize and restore corrupted letters.
        The network learns common letter patterns and can recover them from noisy inputs.
        """)
        
        patterns = st.session_state.demo_data['letter_patterns']
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("📚 Training Patterns")
            
            # Show letter patterns
            for letter, pattern in patterns.items():
                st.write(f"**Letter {letter}:**")
                
                # Create visual representation
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(pattern, cmap='RdYlBu', vmin=-1, vmax=1)
                ax.set_title(f"Letter {letter}", fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add grid
                for i in range(6):
                    ax.axhline(i-0.5, color='black', linewidth=0.5)
                    ax.axvline(i-0.5, color='black', linewidth=0.5)
                
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("🧠 Network Training")
            
            if st.button("Train Letter Recognition Network", type="primary"):
                with st.spinner("Training network..."):
                    # Flatten patterns
                    pattern_array = np.array([p.flatten() for p in patterns.values()])
                    
                    # Create and train network
                    network = HopfieldNetwork(25, "LetterNet")
                    training_stats = network.train_hebbian(pattern_array)
                    
                    st.session_state.networks["LetterNet"] = network
                    st.session_state.current_network = network
                    
                    st.success("Training completed!")
                    
                    # Show training stats
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Patterns Stored", training_stats['n_patterns'])
                        st.metric("Training Time", f"{training_stats['training_time']:.3f}s")
                    with col_b:
                        st.metric("Capacity Ratio", f"{training_stats['capacity_ratio']:.2f}")
                        st.metric("Weight Range", 
                                 f"[{training_stats['weight_stats']['min']:.2f}, {training_stats['weight_stats']['max']:.2f}]")
        
        with col3:
            st.subheader("🔍 Pattern Recognition")
            
            if "LetterNet" in st.session_state.networks:
                network = st.session_state.networks["LetterNet"]
                
                # Select letter to corrupt
                selected_letter = st.selectbox("Select Letter to Test", list(patterns.keys()))
                
                # Corruption settings
                st.write("**Corruption Settings:**")
                noise_type = st.radio("Noise Type", ["flip", "mask", "gaussian"])
                custom_noise = st.slider("Noise Level", 0.0, 0.5, noise_level, 0.05)
                
                if st.button("Test Recognition"):
                    original_pattern = patterns[selected_letter]
                    original_flat = original_pattern.flatten()
                    
                    # Add noise
                    noisy_flat = network.add_noise(original_flat, custom_noise, noise_type)
                    noisy_pattern = noisy_flat.reshape(5, 5)
                    
                    # Recall
                    with st.spinner("Recognizing pattern..."):
                        recall_result = network.recall(noisy_flat, max_iterations=max_iterations, 
                                                     async_update=async_update)
                    
                    recalled_pattern = recall_result['final_states'].reshape(5, 5)
                    
                    # Display results
                    st.write("**Results:**")
                    
                    result_cols = st.columns(3)
                    
                    with result_cols[0]:
                        st.write("*Original*")
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(original_pattern, cmap='RdYlBu', vmin=-1, vmax=1)
                        ax.set_title("Original")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        st.pyplot(fig)
                        plt.close()
                    
                    with result_cols[1]:
                        st.write("*Noisy*")
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(noisy_pattern, cmap='RdYlBu', vmin=-1, vmax=1)
                        ax.set_title("Corrupted")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        st.pyplot(fig)
                        plt.close()
                    
                    with result_cols[2]:
                        st.write("*Recalled*")
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(recalled_pattern, cmap='RdYlBu', vmin=-1, vmax=1)
                        ax.set_title("Recognized")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        st.pyplot(fig)
                        plt.close()
                    
                    # Recognition metrics
                    accuracy = np.mean(recall_result['final_states'] == original_flat)
                    perfect_match = np.array_equal(recall_result['final_states'], original_flat)
                    
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Accuracy", f"{accuracy:.1%}")
                    with metric_cols[1]:
                        st.metric("Perfect Match", "✓" if perfect_match else "✗")
                    with metric_cols[2]:
                        st.metric("Iterations", recall_result['iterations'])
                    with metric_cols[3]:
                        st.metric("Converged", "✓" if recall_result['converged'] else "✗")
                    
                    # Energy plot
                    if network.energy_history:
                        st.subheader("🔋 Energy Evolution")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=network.energy_history,
                            mode='lines+markers',
                            name='Energy',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ))
                        
                        fig.update_layout(
                            title="Network Energy During Recognition",
                            xaxis_title="Iteration",
                            yaxis_title="Energy",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Train the network first to test recognition")
    
    def render_password_recovery(self, max_iterations, noise_level, async_update):
        """Render password recovery demo."""
        st.header("🔐 Password Recovery System")
        
        st.markdown("""
        This demo shows how Hopfield Networks can be used for password recovery systems.
        The network learns common password patterns and can suggest completions for partial passwords.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💾 Password Database")
            
            password_db = st.session_state.demo_data['password_database']
            
            # Show password database
            st.write("**Training Passwords:**")
            for i, pwd in enumerate(password_db, 1):
                st.write(f"{i}. `{pwd}`")
            
            # Training section
            st.subheader("🏋️ Training")
            
            max_length = st.slider("Max Password Length", 8, 16, 12)
            charset = st.text_input("Character Set", "abcdefghijklmnopqrstuvwxyz0123456789")
            
            if st.button("Train Password Recovery System", type="primary"):
                with st.spinner("Training password recovery system..."):
                    recovery = PasswordRecovery()
                    recovery.setup_encoding(charset)
                    
                    training_stats = recovery.train_on_passwords(password_db, max_length)
                    
                    # Store in session
                    st.session_state.password_recovery = recovery
                    
                    st.success("Password recovery system trained!")
                    st.json(training_stats)
        
        with col2:
            st.subheader("🔍 Password Recovery")
            
            if 'password_recovery' in st.session_state:
                recovery = st.session_state.password_recovery
                
                st.write("**Enter partial password:**")
                st.write("Use `*` for unknown characters")
                
                partial_pwd = st.text_input("Partial Password", "p***word***", 
                                          help="Example: p***word123, admin****, user****")
                
                if st.button("Recover Password"):
                    if partial_pwd:
                        with st.spinner("Recovering password..."):
                            # Parse known positions
                            known_positions = []
                            for i, char in enumerate(partial_pwd):
                                if char != '*':
                                    known_positions.append(i)
                            
                            # Recover passwords
                            candidates = recovery.recover_password(
                                partial_pwd, known_positions, max_length
                            )
                            
                            st.subheader("🎯 Recovery Results")
                            
                            if candidates:
                                st.write("**Suggested passwords (ranked by confidence):**")
                                
                                for i, (password, confidence) in enumerate(candidates[:5], 1):
                                    # Check if it matches known passwords
                                    match_indicator = ""
                                    if password.strip() in password_db:
                                        match_indicator = " ✅ (Known password)"
                                    
                                    st.write(f"{i}. `{password}` (confidence: {confidence:.3f}){match_indicator}")
                                
                                # Show best match details
                                best_pwd, best_conf = candidates[0]
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Best Match", best_pwd)
                                with col_b:
                                    st.metric("Confidence", f"{best_conf:.3f}")
                                with col_c:
                                    is_known = best_pwd.strip() in password_db
                                    st.metric("In Database", "✓" if is_known else "✗")
                            
                            else:
                                st.error("No password candidates found")
                    
                    else:
                        st.warning("Please enter a partial password")
            
            else:
                st.info("Train the password recovery system first")
        
        # Security note
        st.markdown("---")
        st.warning("""
        **Security Note:** This is a demonstration only. In practice, password recovery systems 
        should use proper cryptographic methods and never store passwords in plain text.
        """)
    
    def render_tsp_solver(self, max_iterations, noise_level, async_update):
        """Render TSP solver demo."""
        st.header("🗺️ Route Optimization (Traveling Salesman Problem)")
        
        st.markdown("""
        This demo shows how Hopfield Networks can solve optimization problems like the 
        Traveling Salesman Problem (TSP). The network finds the shortest route visiting all cities.
        """)
        
        tsp_data = st.session_state.demo_data['tsp_cities']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏙️ Cities")
            
            # Show city information
            city_df = pd.DataFrame({
                'City': tsp_data['names'],
                'X': tsp_data['coordinates'][:, 0],
                'Y': tsp_data['coordinates'][:, 1]
            })
            
            st.dataframe(city_df, use_container_width=True)
            
            # Visualize cities
            st.subheader("📍 City Map")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=tsp_data['coordinates'][:, 0],
                y=tsp_data['coordinates'][:, 1],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=tsp_data['names'],
                textposition='top center',
                name='Cities'
            ))
            
            fig.update_layout(
                title="City Locations",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🧮 TSP Solver")
            
            # Distance matrix
            st.write("**Distance Matrix:**")
            distance_df = pd.DataFrame(
                tsp_data['distance_matrix'],
                index=tsp_data['names'],
                columns=tsp_data['names']
            )
            st.dataframe(distance_df.round(2), use_container_width=True)
            
            # Solver parameters
            tsp_iterations = st.slider("TSP Max Iterations", 100, 2000, 1000, 100)
            
            if st.button("Solve TSP", type="primary"):
                with st.spinner("Solving Traveling Salesman Problem..."):
                    solver = OptimizationSolver()
                    
                    start_time = time.time()
                    result = solver.traveling_salesman(
                        tsp_data['distance_matrix'], 
                        max_iterations=tsp_iterations
                    )
                    solve_time = time.time() - start_time
                    
                    st.subheader("🎯 Solution")
                    
                    if result['valid_solution']:
                        st.success("✅ Valid solution found!")
                        
                        # Show solution metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Total Distance", f"{result['total_distance']:.2f}")
                        with metric_cols[1]:
                            st.metric("Solve Time", f"{solve_time:.2f}s")
                        with metric_cols[2]:
                            st.metric("Iterations", result['convergence_info']['iterations'])
                        with metric_cols[3]:
                            st.metric("Converged", "✓" if result['convergence_info']['converged'] else "✗")
                        
                        # Show tour
                        tour = result['tour']
                        tour_names = [tsp_data['names'][i] for i in tour]
                        tour_str = " → ".join(tour_names) + f" → {tour_names[0]}"
                        
                        st.write("**Optimal Tour:**")
                        st.write(tour_str)
                        
                        # Visualize solution
                        fig = go.Figure()
                        
                        # Add cities
                        fig.add_trace(go.Scatter(
                            x=tsp_data['coordinates'][:, 0],
                            y=tsp_data['coordinates'][:, 1],
                            mode='markers+text',
                            marker=dict(size=15, color='red'),
                            text=tsp_data['names'],
                            textposition='top center',
                            name='Cities'
                        ))
                        
                        # Add tour path
                        tour_coords = tsp_data['coordinates'][tour]
                        tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the loop
                        
                        fig.add_trace(go.Scatter(
                            x=tour_coords[:, 0],
                            y=tour_coords[:, 1],
                            mode='lines',
                            line=dict(color='blue', width=3),
                            name='Optimal Route'
                        ))
                        
                        fig.update_layout(
                            title=f"TSP Solution (Distance: {result['total_distance']:.2f})",
                            xaxis_title="X Coordinate",
                            yaxis_title="Y Coordinate",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("❌ No valid solution found")
                        st.write("Try increasing the number of iterations or adjusting parameters.")
                        
                        # Show partial results
                        st.write("**Convergence Info:**")
                        st.json(result['convergence_info'])
    
    def render_text_completion(self, max_iterations, noise_level, async_update):
        """Render text completion demo."""
        st.header("📝 Text Pattern Completion")
        
        st.markdown("""
        This demo shows how Hopfield Networks can complete text patterns. 
        The network learns common text sequences and can fill in missing parts.
        """)
        
        text_sequences = st.session_state.demo_data['text_sequences']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📚 Training Sequences")
            
            st.write("**Text patterns for training:**")
            for i, seq in enumerate(text_sequences, 1):
                st.write(f"{i}. `{seq}`")
            
            # Training section
            if st.button("Train Text Completion System", type="primary"):
                with st.spinner("Training text completion system..."):
                    completer = PatternCompletion("text")
                    training_stats = completer.train_sequence_patterns(text_sequences)
                    
                    # Store in session
                    st.session_state.text_completer = completer
                    
                    st.success("Text completion system trained!")
                    
                    # Show vocabulary
                    st.write("**Learned Vocabulary:**")
                    vocab_chars = list(completer.vocab.keys())
                    st.write(" ".join(f"`{char}`" for char in sorted(vocab_chars)))
                    
                    st.json(training_stats)
        
        with col2:
            st.subheader("✍️ Text Completion")
            
            if 'text_completer' in st.session_state:
                completer = st.session_state.text_completer
                
                st.write("**Enter partial text:**")
                st.write("Use `*` for unknown characters")
                
                # Predefined examples
                example_texts = [
                    "ARTIFICIAL INTEL*******",
                    "***URAL NETWORK SYSTEMS",
                    "MACHINE ****NING MODELS",
                    "HOPFIELD NET**** DEMO"
                ]
                
                selected_example = st.selectbox("Select Example", ["Custom"] + example_texts)
                
                if selected_example == "Custom":
                    partial_text = st.text_input("Partial Text", "NEURAL NET****")
                else:
                    partial_text = selected_example
                    st.write(f"Selected: `{partial_text}`")
                
                if st.button("Complete Text"):
                    if partial_text:
                        with st.spinner("Completing text..."):
                            # Parse known positions
                            known_positions = []
                            for i, char in enumerate(partial_text):
                                if char != '*':
                                    known_positions.append(i)
                            
                            # Complete text
                            completions = completer.complete_pattern(
                                partial_text, known_positions
                            )
                            
                            st.subheader("🎯 Completion Results")
                            
                            if completions:
                                st.write("**Suggested completions:**")
                                
                                for i, completion in enumerate(completions[:5], 1):
                                    # Check if it matches known sequences
                                    match_indicator = ""
                                    if completion.strip() in text_sequences:
                                        match_indicator = " ✅ (Training sequence)"
                                    
                                    st.write(f"{i}. `{completion}`{match_indicator}")
                                
                                # Show completion analysis
                                best_completion = completions[0]
                                
                                st.subheader("📊 Analysis")
                                
                                analysis_cols = st.columns(3)
                                with analysis_cols[0]:
                                    st.metric("Best Completion", best_completion)
                                with analysis_cols[1]:
                                    known_chars = len([c for c in partial_text if c != '*'])
                                    completion_ratio = known_chars / len(partial_text)
                                    st.metric("Known Characters", f"{completion_ratio:.1%}")
                                with analysis_cols[2]:
                                    is_training = best_completion.strip() in text_sequences
                                    st.metric("From Training", "✓" if is_training else "✗")
                                
                                # Character-by-character comparison
                                st.subheader("🔍 Character Analysis")
                                
                                char_data = []
                                for i, (orig, comp) in enumerate(zip(partial_text, best_completion)):
                                    char_data.append({
                                        'Position': i,
                                        'Input': orig if orig != '*' else '?',
                                        'Output': comp,
                                        'Status': 'Known' if orig != '*' else 'Predicted'
                                    })
                                
                                char_df = pd.DataFrame(char_data)
                                st.dataframe(char_df, use_container_width=True)
                            
                            else:
                                st.error("No completions found")
                    
                    else:
                        st.warning("Please enter partial text")
            
            else:
                st.info("Train the text completion system first")
    
    def render_custom_network(self, max_iterations, noise_level, async_update):
        """Render custom network builder."""
        st.header("🎯 Custom Network Builder")
        
        st.markdown("""
        Build and experiment with custom Hopfield Networks. Define your own patterns,
        test different configurations, and analyze network behavior.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🏗️ Network Configuration")
            
            with st.form("custom_network"):
                name = st.text_input("Network Name", "CustomNet")
                size = st.number_input("Network Size", 5, 200, 25, 5)
                
                st.write("**Pattern Definition:**")
                pattern_method = st.radio(
                    "Pattern Source",
                    ["Random Patterns", "Manual Entry", "Upload File"]
                )
                
                if pattern_method == "Random Patterns":
                    n_patterns = st.number_input("Number of Patterns", 1, 20, 3)
                    pattern_bias = st.slider("Pattern Bias (towards +1)", 0.0, 1.0, 0.5, 0.1)
                
                elif pattern_method == "Manual Entry":
                    st.write("Enter patterns as comma-separated values (1 or -1):")
                    pattern_text = st.text_area(
                        "Patterns",
                        "1,1,-1,-1,1\n-1,1,1,-1,-1\n1,-1,1,-1,1",
                        help="Each line is a pattern. Use 1 and -1 separated by commas."
                    )
                
                training_method = st.selectbox(
                    "Training Method",
                    ["Hebbian Learning", "Pseudoinverse Rule"]
                )
                
                submitted = st.form_submit_button("Create Custom Network", type="primary")
                
                if submitted:
                    with st.spinner("Creating custom network..."):
                        # Create network
                        network = HopfieldNetwork(size, name)
                        
                        # Generate patterns
                        if pattern_method == "Random Patterns":
                            patterns = np.random.choice(
                                [-1, 1], 
                                size=(n_patterns, size),
                                p=[1-pattern_bias, pattern_bias]
                            )
                        
                        elif pattern_method == "Manual Entry":
                            try:
                                pattern_lines = pattern_text.strip().split('\n')
                                patterns = []
                                
                                for line in pattern_lines:
                                    if line.strip():
                                        pattern = [int(x.strip()) for x in line.split(',')]
                                        if len(pattern) != size:
                                            st.error(f"Pattern length {len(pattern)} doesn't match network size {size}")
                                            st.stop()
                                        patterns.append(pattern)
                                
                                patterns = np.array(patterns)
                                
                            except ValueError as e:
                                st.error(f"Error parsing patterns: {e}")
                                st.stop()
                        
                        # Train network
                        if training_method == "Hebbian Learning":
                            training_stats = network.train_hebbian(patterns)
                        else:
                            training_stats = network.train_pseudoinverse(patterns)
                        
                        # Store network
                        st.session_state.networks[name] = network
                        st.session_state.current_network = network
                        
                        st.success(f"Created custom network '{name}'!")
                        st.json(training_stats)
        
        with col2:
            st.subheader("🧪 Network Testing")
            
            if st.session_state.current_network:
                network = st.session_state.current_network
                
                st.write(f"**Testing Network: {network.name}**")
                
                # Pattern recall test
                if network.patterns is not None and len(network.patterns) > 0:
                    st.write("**Pattern Recall Test:**")
                    
                    pattern_idx = st.selectbox(
                        "Select Pattern to Test",
                        range(len(network.patterns)),
                        format_func=lambda x: f"Pattern {x+1}"
                    )
                    
                    test_noise = st.slider("Test Noise Level", 0.0, 0.5, 0.2, 0.05)
                    test_noise_type = st.selectbox("Noise Type", ["flip", "gaussian", "mask"])
                    
                    if st.button("Test Pattern Recall"):
                        original = network.patterns[pattern_idx]
                        noisy = network.add_noise(original, test_noise, test_noise_type)
                        
                        with st.spinner("Testing recall..."):
                            recall_result = network.recall(
                                noisy, 
                                max_iterations=max_iterations,
                                async_update=async_update
                            )
                        
                        # Show results
                        st.write("**Recall Results:**")
                        
                        result_cols = st.columns(4)
                        with result_cols[0]:
                            accuracy = np.mean(recall_result['final_states'] == original)
                            st.metric("Accuracy", f"{accuracy:.1%}")
                        with result_cols[1]:
                            perfect = np.array_equal(recall_result['final_states'], original)
                            st.metric("Perfect Recall", "✓" if perfect else "✗")
                        with result_cols[2]:
                            st.metric("Iterations", recall_result['iterations'])
                        with result_cols[3]:
                            st.metric("Converged", "✓" if recall_result['converged'] else "✗")
                        
                        # Pattern comparison
                        if network.n_neurons <= 25:  # Only for small patterns
                            st.write("**Pattern Comparison:**")
                            
                            comparison_cols = st.columns(3)
                            
                            with comparison_cols[0]:
                                st.write("*Original*")
                                if network.n_neurons == 25:  # 5x5 grid
                                    pattern_2d = original.reshape(5, 5)
                                    fig, ax = plt.subplots(figsize=(2, 2))
                                    ax.imshow(pattern_2d, cmap='RdYlBu', vmin=-1, vmax=1)
                                    ax.set_title("Original")
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.write(" ".join(['+' if x > 0 else '-' for x in original]))
                            
                            with comparison_cols[1]:
                                st.write("*Noisy*")
                                if network.n_neurons == 25:  # 5x5 grid
                                    noisy_2d = noisy.reshape(5, 5)
                                    fig, ax = plt.subplots(figsize=(2, 2))
                                    ax.imshow(noisy_2d, cmap='RdYlBu', vmin=-1, vmax=1)
                                    ax.set_title("Noisy")
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.write(" ".join(['+' if x > 0 else '-' for x in noisy]))
                            
                            with comparison_cols[2]:
                                st.write("*Recalled*")
                                if network.n_neurons == 25:  # 5x5 grid
                                    recalled_2d = recall_result['final_states'].reshape(5, 5)
                                    fig, ax = plt.subplots(figsize=(2, 2))
                                    ax.imshow(recalled_2d, cmap='RdYlBu', vmin=-1, vmax=1)
                                    ax.set_title("Recalled")
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    st.pyplot(fig)
                                    plt.close()
                                else:
                                    st.write(" ".join(['+' if x > 0 else '-' for x in recall_result['final_states']]))
                        
                        # Energy evolution
                        if network.energy_history:
                            st.write("**Energy Evolution:**")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=network.energy_history,
                                mode='lines+markers',
                                name='Energy'
                            ))
                            
                            fig.update_layout(
                                title="Energy During Recall",
                                xaxis_title="Iteration",
                                yaxis_title="Energy",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("No patterns stored in current network")
            
            else:
                st.info("Create a custom network to start testing")
    
    def run(self):
        """Run the demo application."""
        self.render_header()
        
        # Get settings from sidebar
        app_mode, max_iterations, noise_level, async_update = self.render_sidebar()
        
        # Route to appropriate demo
        if app_mode == "📊 Network Analysis Dashboard":
            self.render_network_dashboard()
        elif app_mode == "🔤 Letter Recognition":
            self.render_letter_recognition(max_iterations, noise_level, async_update)
        elif app_mode == "🔐 Password Recovery":
            self.render_password_recovery(max_iterations, noise_level, async_update)
        elif app_mode == "🗺️ Route Optimization (TSP)":
            self.render_tsp_solver(max_iterations, noise_level, async_update)
        elif app_mode == "📝 Text Completion":
            self.render_text_completion(max_iterations, noise_level, async_update)
        elif app_mode == "🎯 Custom Network Builder":
            self.render_custom_network(max_iterations, noise_level, async_update)


if __name__ == "__main__":
    demo = HopfieldDemo()
    demo.run()