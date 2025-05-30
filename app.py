import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from hopfield_network import HopfieldNetwork
import time


st.set_page_config(
    page_title="Hopfield Network Visualizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Hopfield Network Visualizer")
st.markdown("Interactive visualization of Hopfield Networks for associative memory")

# Sidebar configuration
with st.sidebar:
    st.header("Network Configuration")
    
    mode = st.radio("Mode", ["Manual Connections", "Pattern Recognition"])
    
    if mode == "Manual Connections":
        n_neurons = st.slider("Number of Neurons", 3, 20, 4)
        st.subheader("Add Connections")
        
        if 'connections' not in st.session_state:
            st.session_state.connections = []
            
        col1, col2, col3 = st.columns(3)
        with col1:
            neuron_i = st.number_input("Neuron i", 0, n_neurons-1, 0)
        with col2:
            neuron_j = st.number_input("Neuron j", 0, n_neurons-1, 1)
        with col3:
            weight = st.number_input("Weight", -10.0, 10.0, 1.0, 0.1)
            
        if st.button("Add Connection"):
            if neuron_i != neuron_j:
                st.session_state.connections.append((neuron_i, neuron_j, weight))
                
        if st.button("Clear All Connections"):
            st.session_state.connections = []
            
        if st.session_state.connections:
            st.subheader("Current Connections")
            for i, (ni, nj, w) in enumerate(st.session_state.connections):
                st.text(f"{ni} ↔ {nj}: {w:+.1f}")
                
    else:  # Pattern Recognition mode
        pattern_size = st.slider("Pattern Size", 9, 100, 25)
        grid_size = int(np.sqrt(pattern_size))
        if grid_size * grid_size != pattern_size:
            pattern_size = grid_size * grid_size
            st.info(f"Adjusted to {pattern_size} for square grid")
            
        n_patterns = st.slider("Number of Patterns to Store", 1, 5, 2)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)
        
    update_method = st.radio("Update Method", ["Asynchronous", "Synchronous"])
    max_iterations = st.slider("Max Iterations", 10, 500, 100)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Hopfield Networks are recurrent neural networks that serve as 
    content-addressable memory systems. They can store patterns and 
    retrieve them from partial or noisy inputs.
    """)

# Main content area
if mode == "Manual Connections":
    if st.session_state.connections:
        network = HopfieldNetwork(n_neurons)
        network.set_weights(st.session_state.connections)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Visualization")
            
            # Create network graph
            fig = go.Figure()
            
            # Calculate node positions in a circle
            angles = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            # Add edges
            for i, j, weight in st.session_state.connections:
                color = 'green' if weight > 0 else 'red'
                width = abs(weight)
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[j]],
                    y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False,
                    hovertemplate=f"Connection {i}-{j}: {weight}<extra></extra>"
                ))
                
            # Add nodes
            node_colors = ['blue' if s > 0 else 'orange' for s in network.states]
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(size=30, color=node_colors),
                text=[str(i) for i in range(n_neurons)],
                textposition="middle center",
                textfont=dict(color='white', size=14),
                showlegend=False,
                hovertemplate="Neuron %{text}: State %{marker.color}<extra></extra>"
            ))
            
            fig.update_layout(
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Energy Landscape")
            
            if st.button("Run Network"):
                initial_energy = network.energy()
                if update_method == "Asynchronous":
                    iterations = network.update_async(max_iterations)
                else:
                    for i in range(max_iterations):
                        if not network.update_sync():
                            iterations = i + 1
                            break
                    else:
                        iterations = max_iterations
                        
                final_energy = network.energy()
                
                st.success(f"Converged in {iterations} iterations")
                st.metric("Initial Energy", f"{initial_energy:.2f}")
                st.metric("Final Energy", f"{final_energy:.2f}", 
                         f"{final_energy - initial_energy:.2f}")
                
                # Plot energy history
                if network.energy_history:
                    fig_energy = go.Figure()
                    fig_energy.add_trace(go.Scatter(
                        y=network.energy_history,
                        mode='lines+markers',
                        name='Energy'
                    ))
                    fig_energy.update_layout(
                        xaxis_title="Iteration",
                        yaxis_title="Energy",
                        height=300
                    )
                    st.plotly_chart(fig_energy, use_container_width=True)
                    
            # State display
            st.subheader("Neuron States")
            states_display = " ".join([f"{i}:{'+' if s > 0 else '-'}" 
                                      for i, s in enumerate(network.states)])
            st.code(states_display)
            
else:  # Pattern Recognition mode
    st.subheader("Pattern Storage and Recall")
    
    grid_size = int(np.sqrt(pattern_size))
    
    # Create or load patterns
    if 'stored_patterns' not in st.session_state:
        st.session_state.stored_patterns = []
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Draw Pattern")
        st.markdown("Click cells to toggle")
        
        if 'current_pattern' not in st.session_state:
            st.session_state.current_pattern = np.ones((grid_size, grid_size))
            
        # Create clickable grid with better handling
        grid_container = st.container()
        with grid_container:
            for i in range(grid_size):
                cols = st.columns(grid_size)
                for j in range(grid_size):
                    with cols[j]:
                        if st.button("⬜" if st.session_state.current_pattern[i, j] > 0 else "⬛", 
                                       key=f"cell_{i}_{j}", 
                                       use_container_width=True):
                            st.session_state.current_pattern[i, j] *= -1
                            st.rerun()
                    
        if st.button("Store Pattern"):
            if len(st.session_state.stored_patterns) < n_patterns:
                st.session_state.stored_patterns.append(
                    st.session_state.current_pattern.flatten().copy()
                )
                st.success(f"Pattern {len(st.session_state.stored_patterns)} stored!")
            else:
                st.error(f"Maximum {n_patterns} patterns can be stored")
                
        if st.button("Clear Pattern"):
            st.session_state.current_pattern = np.ones((grid_size, grid_size))
            st.rerun()
            
        if st.button("Random Pattern"):
            st.session_state.current_pattern = np.random.choice(
                [-1, 1], size=(grid_size, grid_size)
            )
            st.rerun()
            
    with col2:
        st.markdown("### Stored Patterns")
        if st.session_state.stored_patterns:
            for idx, pattern in enumerate(st.session_state.stored_patterns):
                st.markdown(f"Pattern {idx + 1}")
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(pattern.reshape(grid_size, grid_size), 
                         cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                plt.close()
        else:
            st.info("No patterns stored yet")
            
        if st.button("Clear All Stored Patterns"):
            st.session_state.stored_patterns = []
            st.rerun()
            
    with col3:
        st.markdown("### Pattern Recall")
        if st.session_state.stored_patterns:
            # Train network
            network = HopfieldNetwork(pattern_size)
            patterns_array = np.array(st.session_state.stored_patterns)
            network.train(patterns_array)
            
            # Add noise to current pattern
            noisy_pattern = network.add_noise(
                st.session_state.current_pattern.flatten(), 
                noise_level
            )
            
            st.markdown("Noisy Input")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(noisy_pattern.reshape(grid_size, grid_size), 
                     cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            if st.button("Recall Pattern"):
                recalled = network.recall(
                    noisy_pattern,
                    async_update=(update_method == "Asynchronous"),
                    max_iterations=max_iterations
                )
                
                st.markdown("Recalled Pattern")
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(recalled.reshape(grid_size, grid_size), 
                         cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
                # Check which stored pattern it matches
                for idx, stored in enumerate(st.session_state.stored_patterns):
                    if np.array_equal(recalled, stored):
                        st.success(f"Recalled Pattern {idx + 1}")
                        break
                else:
                    st.warning("Recalled pattern doesn't match any stored pattern")
                    
                # Show accuracy
                if len(st.session_state.stored_patterns) > 0:
                    best_match = max(
                        [(np.mean(recalled == p), i) 
                         for i, p in enumerate(st.session_state.stored_patterns)]
                    )
                    st.metric("Best Match Accuracy", 
                             f"{best_match[0]:.1%} with Pattern {best_match[1]+1}")
        else:
            st.info("Store some patterns first")
            
# Footer
st.markdown("---")
st.markdown("### How it works")
st.markdown("""
- **Manual Connections**: Define custom network topology and watch it converge to stable states
- **Pattern Recognition**: Store patterns and recall them from noisy inputs
- **Energy Minimization**: The network seeks states that minimize its energy function
- **Asynchronous vs Synchronous**: Different update strategies affect convergence behavior
""")

# Performance metrics
with st.expander("Performance Metrics"):
    st.markdown("""
    This implementation uses NumPy for fast vectorized operations:
    - Pattern storage: O(n²) per pattern
    - Recall: O(n²) per iteration
    - Memory: O(n²) for weight matrix
    
    The network can handle patterns up to 10,000 neurons efficiently.
    """)