#!/usr/bin/env python3
"""
🧠 Neural Network Explorer - For High School Students
Learn AI through interactive experiments!
"""

import streamlit as st
import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion, ImageRestoration
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page setup
st.set_page_config(
    page_title="🧠 Neural Network Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, modern design
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    .experiment-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
    }
    
    .pattern-grid {
        display: grid;
        grid-gap: 2px;
        justify-content: center;
        margin: 10px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Neural Network Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### *Discover how AI learns and remembers - Interactive experiments for students*")
    
    # Sidebar navigation
    st.sidebar.markdown("## 🎯 Choose Your Experiment")
    experiment = st.sidebar.selectbox(
        "Select an experiment:",
        [
            "🎨 Pattern Memory Lab",
            "🔐 Password Recovery AI", 
            "🧩 Sequence Prediction",
            "📊 Network Analysis",
            "🖼️ Image Restoration",
            "📚 How It Works"
        ]
    )
    
    # Display selected experiment
    if experiment == "🎨 Pattern Memory Lab":
        pattern_memory_lab()
    elif experiment == "🔐 Password Recovery AI":
        password_recovery_lab()
    elif experiment == "🧩 Sequence Prediction":
        sequence_prediction_lab()
    elif experiment == "📊 Network Analysis":
        network_analysis_lab()
    elif experiment == "🖼️ Image Restoration":
        image_restoration_lab()
    else:
        how_it_works()

def pattern_memory_lab():
    """Interactive pattern memory experiment"""
    st.header("🎨 Pattern Memory Lab")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Experiment Goal:</strong> See how neural networks store and recall patterns, even when they're corrupted!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Step 1: Create Patterns")
        
        # Pattern size selector
        pattern_size = st.selectbox("Pattern size:", [5, 7, 9], index=0)
        
        # Initialize pattern grid
        if 'pattern_grid' not in st.session_state or st.session_state.get('pattern_size') != pattern_size:
            st.session_state.pattern_grid = np.ones((pattern_size, pattern_size)) * -1
            st.session_state.pattern_size = pattern_size
            st.session_state.stored_patterns = []
        
        # Display clickable grid
        st.write("Click cells to draw (⬛ = ON, ⬜ = OFF):")
        
        # Create grid layout
        grid_container = st.container()
        with grid_container:
            for i in range(pattern_size):
                cols = st.columns(pattern_size)
                for j in range(pattern_size):
                    if cols[j].button(
                        "⬛" if st.session_state.pattern_grid[i,j] == 1 else "⬜", 
                        key=f"grid_{i}_{j}",
                        help="Click to toggle"
                    ):
                        st.session_state.pattern_grid[i,j] *= -1
                        st.rerun()
        
        # Pattern controls
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear", key="clear_pattern"):
                st.session_state.pattern_grid = np.ones((pattern_size, pattern_size)) * -1
                st.rerun()
        
        with col_b:
            if st.button("💾 Store Pattern", key="store_pattern"):
                st.session_state.stored_patterns.append(st.session_state.pattern_grid.copy())
                st.success(f"Pattern {len(st.session_state.stored_patterns)} stored!")
        
        # Preset patterns
        st.write("Or use presets:")
        if st.button("➕ Cross", key="preset_cross"):
            cross = create_cross_pattern(pattern_size)
            st.session_state.pattern_grid = cross
            st.rerun()
        
        if st.button("⬛ Square", key="preset_square"):
            square = create_square_pattern(pattern_size)
            st.session_state.pattern_grid = square
            st.rerun()
    
    with col2:
        st.subheader("Step 2: Train Network")
        
        if len(st.session_state.stored_patterns) > 0:
            st.write(f"**Stored patterns:** {len(st.session_state.stored_patterns)}")
            
            # Display stored patterns
            for i, pattern in enumerate(st.session_state.stored_patterns):
                st.write(f"Pattern {i+1}:")
                display_pattern_small(pattern)
            
            if st.button("🧠 Train AI Network", key="train_network"):
                with st.spinner("Training neural network..."):
                    # Create and train network
                    network = HopfieldNetwork(n_neurons=pattern_size*pattern_size)
                    patterns_flat = [p.flatten() for p in st.session_state.stored_patterns]
                    network.train_hebbian(np.array(patterns_flat))
                    st.session_state.trained_network = network
                    time.sleep(1)
                
                st.markdown("""
                <div class="success-box">
                ✅ <strong>Network trained!</strong> The AI has memorized your patterns.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Create and store some patterns first!")
    
    with col3:
        st.subheader("Step 3: Test Memory")
        
        if 'trained_network' in st.session_state:
            # Noise controls
            noise_level = st.slider("Corruption level:", 0, 50, 20, help="Percentage of pixels to flip")
            
            if st.button("🌪️ Add Noise & Test", key="add_noise"):
                # Pick random stored pattern
                original_idx = random.randint(0, len(st.session_state.stored_patterns) - 1)
                original = st.session_state.stored_patterns[original_idx]
                
                # Add noise
                noisy = add_noise_to_pattern(original, noise_level / 100)
                
                # Store for recall
                st.session_state.original_pattern = original
                st.session_state.noisy_pattern = noisy
                st.session_state.test_idx = original_idx
            
            if 'noisy_pattern' in st.session_state:
                st.write(f"**Original Pattern {st.session_state.test_idx + 1}:**")
                display_pattern_small(st.session_state.original_pattern)
                
                st.write("**Corrupted Version:**")
                display_pattern_small(st.session_state.noisy_pattern)
                
                if st.button("🔄 AI Recall", key="recall_pattern"):
                    with st.spinner("AI is thinking..."):
                        result = st.session_state.trained_network.recall(
                            st.session_state.noisy_pattern.flatten(), 
                            max_iterations=20
                        )
                        st.session_state.recalled_pattern = result['final_states'].reshape(st.session_state.pattern_size, st.session_state.pattern_size)
                        time.sleep(1)
                
                if 'recalled_pattern' in st.session_state:
                    st.write("**AI Recalled:**")
                    display_pattern_small(st.session_state.recalled_pattern)
                    
                    # Calculate accuracy
                    accuracy = np.mean(st.session_state.recalled_pattern == st.session_state.original_pattern) * 100
                    
                    if accuracy == 100:
                        st.balloons()
                        st.success(f"🎉 Perfect recall! (100% accuracy)")
                    else:
                        st.info(f"🎯 Recall accuracy: {accuracy:.1f}%")
        else:
            st.info("👈 Train the network first!")

def password_recovery_lab():
    """Password recovery experiment"""
    st.header("🔐 Password Recovery AI")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Experiment Goal:</strong> Train an AI to recover corrupted passwords - useful for real security applications!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize password AI
    if 'password_ai' not in st.session_state:
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        # Train with common password patterns
        passwords = [
            "mydog123", "password", "qwerty12", "welcome1",
            "sunshine", "dragon99", "princess", "football",
            "computer", "rainbow7", "butterfly", "superman"
        ]
        recovery.train_on_passwords(passwords, max_length=8)
        st.session_state.password_ai = recovery
        st.session_state.password_db = passwords
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Password Database")
        st.write("The AI knows these passwords:")
        
        # Display password database in a nice format
        for i, pwd in enumerate(st.session_state.password_db):
            st.code(f"{i+1:2d}. {pwd}")
    
    with col2:
        st.subheader("Recovery Test")
        
        # Interactive corruption
        st.write("**Test the AI's memory:**")
        
        # Select test method
        test_method = st.radio(
            "Choose test type:",
            ["🎯 Try examples", "✏️ Custom input"],
            horizontal=True
        )
        
        if test_method == "🎯 Try examples":
            # Pre-made examples
            examples = [
                ("myd___23", "mydog123"),
                ("p___word", "password"),
                ("q___ty12", "qwerty12"),
                ("sun___ne", "sunshine"),
                ("____ball", "football")
            ]
            
            for corrupted, original in examples:
                if st.button(f"Test: {corrupted}", key=f"test_{corrupted}"):
                    with st.spinner("AI analyzing..."):
                        recovered = st.session_state.password_ai.recover_password(corrupted)
                        time.sleep(0.8)
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Corrupted", corrupted)
                    col_b.metric("AI Recovered", recovered)
                    col_c.metric("Original", original)
                    
                    if recovered == original:
                        st.success("✅ Perfect recovery!")
                        st.balloons()
                    else:
                        st.warning("⚠️ Partial recovery - try different corruption")
        
        else:
            # Custom input
            st.write("Enter a corrupted password (use _ for missing characters):")
            custom_input = st.text_input("Corrupted password:", "my___123")
            
            if st.button("🔍 Recover Password", key="custom_recover"):
                if custom_input:
                    with st.spinner("AI working..."):
                        recovered = st.session_state.password_ai.recover_password(custom_input)
                        time.sleep(1)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>AI Recovery Result:</strong><br>
                    Input: <code>{custom_input}</code><br>
                    Output: <code>{recovered}</code>
                    </div>
                    """, unsafe_allow_html=True)

def sequence_prediction_lab():
    """Sequence prediction experiment"""
    st.header("🧩 Sequence Prediction")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Experiment Goal:</strong> See how AI learns patterns in sequences - like autocomplete on steroids!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize sequence AI
    if 'sequence_ai' not in st.session_state:
        completion = PatternCompletion("text")
        
        # Train with various patterns
        sequences = [
            "ABCDABCD",  # Repeating
            "ABCDEFGH",  # Sequential
            "AABBCCDD",  # Paired
            "ABABABAB",  # Alternating
            "AACCAACC",  # Pattern
            "12312312",  # Number pattern (same length)
            "XYZXYZXY",  # Another repeating
        ]
        
        completion.train_sequence_patterns(sequences)
        st.session_state.sequence_ai = completion
        st.session_state.training_sequences = sequences
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Training Data")
        st.write("AI learned these patterns:")
        
        for i, seq in enumerate(st.session_state.training_sequences):
            st.code(f"{i+1}. {seq}")
        
        st.markdown("""
        <div class="info-box">
        💡 <strong>Notice:</strong> Each sequence has a different pattern - 
        repeating, sequential, alternating, etc.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Prediction Test")
        
        # Test modes
        test_mode = st.radio(
            "Test mode:",
            ["🎲 Random tests", "🎯 Guided tests", "✏️ Custom"],
            horizontal=True
        )
        
        if test_mode == "🎲 Random tests":
            if st.button("🎲 Generate Random Test", key="random_test"):
                # Pick random sequence and truncate it
                seq = random.choice(st.session_state.training_sequences)
                truncate_at = random.randint(4, 6)
                partial = seq[:truncate_at] + "_" * (8 - truncate_at)
                
                with st.spinner("AI predicting..."):
                    completed = st.session_state.sequence_ai.complete_sequence(partial)
                    time.sleep(0.8)
                
                st.markdown(f"""
                **Partial:** `{partial}`  
                **AI Prediction:** `{completed}`  
                **Original:** `{seq}`
                """)
                
                if completed == seq:
                    st.success("🎉 Perfect prediction!")
                else:
                    st.info("🤔 AI found a different but valid pattern!")
        
        elif test_mode == "🎯 Guided tests":
            # Specific test cases
            tests = [
                ("ABCD____", "What comes after ABCD?"),
                ("ABAB____", "Can you spot the alternating pattern?"),
                ("AABC____", "Mixed pattern - tricky!"),
                ("1231____", "Number sequence test")
            ]
            
            for partial, description in tests:
                st.write(f"**{description}**")
                if st.button(f"Test: {partial}", key=f"guided_{partial}"):
                    with st.spinner("AI analyzing pattern..."):
                        completed = st.session_state.sequence_ai.complete_sequence(partial)
                        time.sleep(1)
                    
                    st.markdown(f"**Result:** `{completed}`")
        
        else:
            # Custom input
            custom_seq = st.text_input("Enter partial sequence (use _ for missing):", "ABC_____")
            
            if st.button("🔮 Predict", key="custom_predict"):
                if custom_seq and "_" in custom_seq:
                    with st.spinner("AI thinking..."):
                        completed = st.session_state.sequence_ai.complete_sequence(custom_seq)
                        time.sleep(1)
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <strong>AI Prediction:</strong><br>
                    Input: <code>{custom_seq}</code><br>
                    Output: <code>{completed}</code>
                    </div>
                    """, unsafe_allow_html=True)

def network_analysis_lab():
    """Network analysis and visualization"""
    st.header("📊 Network Analysis")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Experiment Goal:</strong> Understand how the neural network actually works under the hood!
    </div>
    """, unsafe_allow_html=True)
    
    # Create sample network for analysis
    if 'analysis_network' not in st.session_state:
        network = HopfieldNetwork(n_neurons=25)
        
        # Create some test patterns
        patterns = [
            create_cross_pattern(5).flatten(),
            create_square_pattern(5).flatten(),
            np.random.choice([-1, 1], 25)  # Random pattern
        ]
        
        network.train_hebbian(np.array(patterns))
        st.session_state.analysis_network = network
        st.session_state.analysis_patterns = patterns
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Network Properties")
        
        network = st.session_state.analysis_network
        
        # Display key metrics
        st.metric("Network Size", f"{network.n_neurons} neurons")
        st.metric("Stored Patterns", len(st.session_state.analysis_patterns))
        st.metric("Connections", f"{network.n_neurons * (network.n_neurons - 1):,}")
        
        # Theoretical capacity
        theoretical_capacity = int(0.138 * network.n_neurons)
        st.metric("Theoretical Capacity", f"~{theoretical_capacity} patterns")
        
        # Weight matrix visualization
        st.subheader("Connection Strengths")
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(network.weights, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title("Weight Matrix (Neuron Connections)")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")
        plt.colorbar(im, ax=ax, label="Connection Strength")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Energy Landscape")
        
        # Show energy calculation
        st.write("**Energy Analysis:**")
        
        for i, pattern in enumerate(st.session_state.analysis_patterns):
            energy = network.energy(pattern)
            st.metric(f"Pattern {i+1} Energy", f"{energy:.2f}")
        
        # Test with noisy patterns
        st.subheader("Noise Tolerance Test")
        
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        if st.button("🧪 Run Noise Test", key="noise_test"):
            results = []
            
            progress_bar = st.progress(0)
            
            for i, noise_level in enumerate(noise_levels):
                accuracies = []
                
                for pattern in st.session_state.analysis_patterns:
                    # Add noise
                    noisy = add_noise_to_pattern(pattern.reshape(5, 5), noise_level).flatten()
                    
                    # Recall
                    result = network.recall(noisy, max_iterations=20)
                    
                    # Calculate accuracy
                    accuracy = np.mean(result['final_states'] == pattern)
                    accuracies.append(accuracy)
                
                avg_accuracy = np.mean(accuracies)
                results.append(avg_accuracy)
                progress_bar.progress((i + 1) / len(noise_levels))
            
            # Plot results
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(noise_levels, results, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel("Noise Level")
            ax.set_ylabel("Recall Accuracy")
            ax.set_title("Network Performance vs Noise")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)
            
            st.info(f"💡 The network maintains good performance up to ~{noise_levels[np.where(np.array(results) > 0.8)[0][-1] if len(np.where(np.array(results) > 0.8)[0]) > 0 else 0]:.1f} noise level!")

def image_restoration_lab():
    """Image restoration experiment"""
    st.header("🖼️ Image Restoration")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Experiment Goal:</strong> Use AI to restore damaged images - like magic photo repair!
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize image restoration
    if 'image_ai' not in st.session_state:
        restorer = ImageRestoration(patch_size=8)
        st.session_state.image_ai = restorer
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        
        # Create a simple test image
        image_type = st.selectbox("Choose image:", ["Checkerboard", "Stripes", "Dots"])
        
        if image_type == "Checkerboard":
            image = create_checkerboard(32)
        elif image_type == "Stripes":
            image = create_stripes(32)
        else:
            image = create_dots(32)
        
        st.image(image, caption="Original", use_column_width=True)
        
        if st.button("🧠 Train AI on this image", key="train_image"):
            with st.spinner("Training on image patches..."):
                st.session_state.image_ai.train_on_image(image)
                time.sleep(1)
            st.success("✅ AI trained!")
    
    with col2:
        st.subheader("Damaged Image")
        
        if 'image_ai' in st.session_state:
            damage_type = st.selectbox("Damage type:", ["Random noise", "Missing blocks", "Scratches"])
            damage_level = st.slider("Damage level:", 0.1, 0.5, 0.2)
            
            if st.button("💥 Damage Image", key="damage_image"):
                damaged = damage_image(image, damage_type, damage_level)
                st.session_state.damaged_image = damaged
            
            if 'damaged_image' in st.session_state:
                st.image(st.session_state.damaged_image, caption="Damaged", use_column_width=True)
    
    with col3:
        st.subheader("AI Restoration")
        
        if 'damaged_image' in st.session_state:
            if st.button("✨ Restore Image", key="restore_image"):
                with st.spinner("AI restoring image..."):
                    restored = st.session_state.image_ai.restore_image(st.session_state.damaged_image)
                    st.session_state.restored_image = restored
                    time.sleep(1.5)
            
            if 'restored_image' in st.session_state:
                st.image(st.session_state.restored_image, caption="AI Restored", use_column_width=True)
                
                # Calculate restoration quality
                mse = np.mean((image - st.session_state.restored_image) ** 2)
                quality = max(0, 100 - mse * 100)
                
                st.metric("Restoration Quality", f"{quality:.1f}%")

def how_it_works():
    """Educational content about how Hopfield networks work"""
    st.header("📚 How Neural Networks Learn")
    
    st.markdown("""
    ## 🧠 What is a Hopfield Network?
    
    A Hopfield Network is like a **digital brain** that can:
    - 📝 **Store memories** (patterns)
    - 🔄 **Recall memories** from partial information  
    - 🛠️ **Fix corrupted data** automatically
    
    ### 🤔 How Does It Work?
    
    Think of it like your brain recognizing a friend:
    1. **Training**: You see your friend many times (storing the pattern)
    2. **Recognition**: Even with sunglasses or a hat, you still recognize them
    3. **Completion**: Your brain "fills in" the missing details
    
    ### 🔬 The Science Behind It
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Neurons & Connections:**
        - Each neuron is either ON (+1) or OFF (-1)
        - Neurons are connected to every other neuron
        - Connection strength = how much neurons influence each other
        
        **Learning (Hebbian Rule):**
        - "Neurons that fire together, wire together"
        - When two neurons are both ON, their connection gets stronger
        - This creates "memory patterns" in the network
        """)
    
    with col2:
        st.markdown("""
        **Recall Process:**
        - Start with corrupted/partial pattern
        - Each neuron looks at its neighbors
        - Neurons flip to match the strongest learned pattern
        - Process repeats until network stabilizes
        
        **Energy Minimization:**
        - Network "rolls downhill" to lowest energy state
        - Stored patterns = energy minima (stable states)
        - Noisy inputs get "pulled" toward nearest stored pattern
        """)
    
    st.markdown("""
    ### 🌟 Real-World Applications
    
    **Image Processing:**
    - Photo restoration and noise removal
    - Medical image enhancement
    - Satellite image analysis
    
    **Data Recovery:**
    - Reconstructing corrupted files
    - Password recovery systems
    - Error correction in communications
    
    **Pattern Recognition:**
    - Handwriting recognition
    - Face recognition systems
    - Voice pattern analysis
    
    **Artificial Intelligence:**
    - Associative memory systems
    - Content-addressable memory
    - Optimization problems
    
    ### 🚀 Try It Yourself!
    
    Use the experiments in this app to:
    1. **Start simple** - try the Pattern Memory Lab
    2. **Experiment** - change noise levels, pattern sizes
    3. **Observe** - how does performance change?
    4. **Learn** - what patterns work best?
    
    Remember: **The best way to learn AI is to play with it!** 🎮
    """)

# Helper functions
def create_cross_pattern(size):
    """Create a cross pattern"""
    pattern = np.ones((size, size)) * -1
    mid = size // 2
    pattern[mid, :] = 1
    pattern[:, mid] = 1
    return pattern

def create_square_pattern(size):
    """Create a square pattern"""
    pattern = np.ones((size, size)) * -1
    pattern[1:-1, 1:-1] = -1
    pattern[0, :] = 1
    pattern[-1, :] = 1
    pattern[:, 0] = 1
    pattern[:, -1] = 1
    return pattern

def add_noise_to_pattern(pattern, noise_level):
    """Add noise to a pattern"""
    noisy = pattern.copy()
    flat = noisy.flatten()
    n_flip = int(len(flat) * noise_level)
    flip_indices = np.random.choice(len(flat), n_flip, replace=False)
    flat[flip_indices] *= -1
    return flat.reshape(pattern.shape)

def display_pattern_small(pattern):
    """Display a pattern as a small grid"""
    display_str = ""
    for row in pattern:
        for val in row:
            display_str += "⬛ " if val == 1 else "⬜ "
        display_str += "\n"
    st.text(display_str)

def create_checkerboard(size):
    """Create checkerboard pattern"""
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                pattern[i, j] = 1
    return pattern

def create_stripes(size):
    """Create stripe pattern"""
    pattern = np.zeros((size, size))
    for i in range(size):
        if i % 4 < 2:
            pattern[i, :] = 1
    return pattern

def create_dots(size):
    """Create dot pattern"""
    pattern = np.zeros((size, size))
    for i in range(2, size-2, 4):
        for j in range(2, size-2, 4):
            pattern[i-1:i+2, j-1:j+2] = 1
    return pattern

def damage_image(image, damage_type, level):
    """Add damage to image"""
    damaged = image.copy()
    
    if damage_type == "Random noise":
        noise_mask = np.random.random(image.shape) < level
        damaged[noise_mask] = 1 - damaged[noise_mask]
    
    elif damage_type == "Missing blocks":
        h, w = image.shape
        n_blocks = int(level * 20)
        for _ in range(n_blocks):
            x = np.random.randint(0, h-4)
            y = np.random.randint(0, w-4)
            damaged[x:x+4, y:y+4] = 0.5
    
    else:  # Scratches
        n_scratches = int(level * 10)
        for _ in range(n_scratches):
            x = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1]-10)
            damaged[x, y:y+10] = 0.5
    
    return damaged

if __name__ == "__main__":
    main()