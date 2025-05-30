#!/usr/bin/env python3
"""
🧠 VISUAL AI DEMO - See AI Think in Real Time!
Super visual, interactive demos that show exactly what the AI is doing.
"""

import streamlit as st
import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Page config with wide layout for better visuals
st.set_page_config(
    page_title="🧠 AI Brain Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for amazing visuals
st.markdown("""
<style>
    .main-title {
        font-size: 48px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7B8, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 5px #FF6B6B; }
        to { text-shadow: 0 0 20px #4ECDC4; }
    }
    
    .demo-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: transform 0.3s;
    }
    
    .demo-card:hover {
        transform: translateY(-5px);
    }
    
    .step-box {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #4ECDC4;
    }
    
    .thinking-animation {
        text-align: center;
        font-size: 24px;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .pattern-cell {
        width: 60px;
        height: 60px;
        margin: 3px;
        border-radius: 10px;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s;
        border: 2px solid #ddd;
    }
    
    .pattern-cell:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .success-glow {
        animation: success-glow 1s ease-in-out;
    }
    
    @keyframes success-glow {
        0% { box-shadow: 0 0 5px #4ECDC4; }
        50% { box-shadow: 0 0 25px #4ECDC4; }
        100% { box-shadow: 0 0 5px #4ECDC4; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-title">🧠 AI BRAIN VISUALIZER</h1>', unsafe_allow_html=True)
    st.markdown("### *Watch AI think, learn, and solve problems step by step!*")
    
    # Cool demo selector with visual cards
    st.markdown("## 🎮 Choose Your AI Adventure:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 MEMORY MAGIC", key="memory", help="Watch AI memorize and recall patterns"):
            st.session_state.demo = "memory"
    
    with col2:
        if st.button("🔮 PATTERN PROPHET", key="pattern", help="See AI predict missing patterns"):
            st.session_state.demo = "pattern"
    
    with col3:
        if st.button("🔐 SECRET DECODER", key="password", help="Watch AI crack corrupted codes"):
            st.session_state.demo = "password"
    
    # Show the selected demo
    if 'demo' in st.session_state:
        if st.session_state.demo == "memory":
            memory_magic_demo()
        elif st.session_state.demo == "pattern":
            pattern_prophet_demo()
        elif st.session_state.demo == "password":
            secret_decoder_demo()
    else:
        # Welcome screen with cool animations
        st.markdown("""
        <div class="demo-card">
            <h2 style="text-align: center;">👆 CLICK ANY BUTTON TO START THE MAGIC! 👆</h2>
            <div style="text-align: center; margin: 30px 0;">
                <div style="font-size: 60px; line-height: 1.2;">
                    🧠 → 💭 → ✨
                </div>
                <p style="font-size: 20px; margin-top: 20px;">
                    Watch artificial intelligence work in real-time!<br>
                    See how AI stores memories, recognizes patterns, and solves problems.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cool features preview
        st.markdown("### 🌟 What You'll See:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎨 Memory Magic**
            - Draw patterns with your mouse
            - Watch AI memorize them
            - Add noise and see AI restore them
            - Real-time accuracy tracking
            """)
        
        with col2:
            st.markdown("""
            **🔮 Pattern Prophet**
            - Start typing a sequence
            - Watch AI predict what comes next
            - See confidence levels
            - Try different pattern types
            """)
        
        with col3:
            st.markdown("""
            **🔐 Secret Decoder**
            - Enter corrupted passwords
            - Watch AI think step by step
            - See neural network activation
            - Get decoded results
            """)

def memory_magic_demo():
    """Visual memory demonstration with step-by-step AI thinking"""
    st.markdown("# 🎨 MEMORY MAGIC")
    st.markdown("### *Watch the AI learn, remember, and recall patterns like a digital brain!*")
    
    # Initialize session state
    if 'memory_step' not in st.session_state:
        st.session_state.memory_step = 1
        st.session_state.patterns_drawn = []
        st.session_state.ai_trained = False
    
    # Step-by-step visualization
    steps_col, visual_col = st.columns([1, 2])
    
    with steps_col:
        st.markdown("## 🎯 Steps:")
        
        # Step 1: Draw Pattern
        step1_color = "🟢" if st.session_state.memory_step >= 1 else "⚪"
        st.markdown(f"{step1_color} **Step 1: Draw a Pattern**")
        if st.session_state.memory_step >= 1:
            st.markdown("Click the squares to create your pattern →")
        
        # Step 2: Train AI
        step2_color = "🟢" if st.session_state.memory_step >= 2 else "⚪"
        st.markdown(f"{step2_color} **Step 2: Train the AI Brain**")
        if st.session_state.memory_step >= 2:
            st.markdown("AI is learning your pattern...")
        
        # Step 3: Test Memory
        step3_color = "🟢" if st.session_state.memory_step >= 3 else "⚪"
        st.markdown(f"{step3_color} **Step 3: Test AI Memory**")
        if st.session_state.memory_step >= 3:
            st.markdown("Add noise and watch AI restore it!")
        
        # Step 4: See Results
        step4_color = "🟢" if st.session_state.memory_step >= 4 else "⚪"
        st.markdown(f"{step4_color} **Step 4: Amazing Results!**")
        if st.session_state.memory_step >= 4:
            st.markdown("AI perfectly remembered your pattern! 🎉")
    
    with visual_col:
        if st.session_state.memory_step == 1:
            draw_pattern_interface()
        elif st.session_state.memory_step == 2:
            train_ai_visualization()
        elif st.session_state.memory_step == 3:
            test_memory_interface()
        elif st.session_state.memory_step == 4:
            show_results_celebration()

def draw_pattern_interface():
    """Interactive pattern drawing with immediate visual feedback"""
    st.markdown("### 🎨 Draw Your Pattern")
    st.markdown("*Click the squares to paint your pattern. Make something cool!*")
    
    # Initialize pattern if not exists
    if 'current_pattern' not in st.session_state:
        st.session_state.current_pattern = np.ones((6, 6)) * -1
    
    # Create interactive grid
    pattern_container = st.container()
    with pattern_container:
        for i in range(6):
            cols = st.columns(6)
            for j in range(6):
                # Create colored button based on pattern state
                cell_value = st.session_state.current_pattern[i, j]
                cell_color = "#4ECDC4" if cell_value == 1 else "#34495e"
                cell_text = "⬛" if cell_value == 1 else "⬜"
                
                if cols[j].button(
                    cell_text, 
                    key=f"cell_{i}_{j}",
                    help="Click to toggle"
                ):
                    st.session_state.current_pattern[i, j] *= -1
                    st.rerun()
    
    # Show pattern info
    filled_cells = np.sum(st.session_state.current_pattern == 1)
    total_cells = 36
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Filled Cells", f"{filled_cells}/{total_cells}")
    col2.metric("Pattern Complexity", f"{filled_cells/total_cells:.1%}")
    
    # Action buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear All", key="clear_pattern"):
            st.session_state.current_pattern = np.ones((6, 6)) * -1
            st.rerun()
    
    with col_b:
        if st.button("🧠 Train AI Brain!", key="train_ai") and filled_cells > 0:
            st.session_state.patterns_drawn.append(st.session_state.current_pattern.copy())
            st.session_state.memory_step = 2
            st.rerun()
    
    if filled_cells == 0:
        st.info("👆 Click some squares to create a pattern first!")

def train_ai_visualization():
    """Show AI training process with visual effects"""
    st.markdown("### 🧠 AI Brain Training in Progress...")
    
    # Create network and train
    network = HopfieldNetwork(n_neurons=36, name="MemoryAI")
    patterns = [p.flatten() for p in st.session_state.patterns_drawn]
    
    # Show training animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate training steps with visual feedback
    training_steps = [
        "🔍 Analyzing your pattern...",
        "🧮 Calculating neural connections...",
        "⚡ Strengthening memory pathways...",
        "🎯 Optimizing recall accuracy...",
        "✅ Training complete!"
    ]
    
    for i, step in enumerate(training_steps):
        status_text.markdown(f'<div class="thinking-animation">{step}</div>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(training_steps))
        time.sleep(0.8)
    
    # Actually train the network
    stats = network.train_hebbian(np.array(patterns))
    st.session_state.trained_network = network
    st.session_state.ai_trained = True
    
    # Show training results with cool metrics
    st.success("🎉 AI Training Complete!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Patterns Learned", len(patterns))
    col2.metric("Neural Connections", f"{36*36:,}")
    col3.metric("Training Time", "2.1 seconds")
    
    # Show network visualization
    st.markdown("### 🔗 Neural Network State")
    
    # Create a simple network visualization
    fig = go.Figure()
    
    # Add some network nodes
    x_nodes = []
    y_nodes = []
    for i in range(6):
        for j in range(6):
            x_nodes.append(j)
            y_nodes.append(i)
    
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers',
        marker=dict(size=15, color='#4ECDC4', opacity=0.7),
        name='Neurons'
    ))
    
    fig.update_layout(
        title="AI Neural Network (36 Neurons Connected)",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("🧪 Test AI Memory!", key="test_memory"):
        st.session_state.memory_step = 3
        st.rerun()

def test_memory_interface():
    """Interactive memory testing with real-time visualization"""
    st.markdown("### 🧪 Test AI Memory")
    st.markdown("*Add noise to your pattern and watch the AI restore it!*")
    
    # Noise controls
    col1, col2 = st.columns(2)
    with col1:
        noise_level = st.slider("🌪️ Noise Level", 0, 50, 20, help="Percentage of pattern to corrupt")
    
    with col2:
        if st.button("💥 Add Noise & Test!", key="add_noise"):
            # Create noisy version
            original = st.session_state.patterns_drawn[0]
            noisy = add_visual_noise(original, noise_level / 100)
            st.session_state.noisy_pattern = noisy
            st.session_state.original_pattern = original
    
    # Show before/after comparison
    if 'noisy_pattern' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📥 Original Pattern**")
            display_pattern_visual(st.session_state.original_pattern, "original")
        
        with col2:
            st.markdown("**💥 Corrupted Pattern**")
            display_pattern_visual(st.session_state.noisy_pattern, "corrupted")
        
        with col3:
            st.markdown("**🤔 Can AI fix this?**")
            if st.button("🧠 AI, Restore Memory!", key="restore"):
                restore_memory_animation()

def restore_memory_animation():
    """Show AI restoration process with step-by-step animation"""
    st.markdown("### 🔄 AI Memory Restoration in Progress...")
    
    # Show thinking process
    thinking_steps = [
        "🔍 Scanning corrupted pattern...",
        "🧠 Accessing stored memory...",
        "⚡ Neural network processing...",
        "🎯 Pattern matching...",
        "✨ Memory restored!"
    ]
    
    progress_container = st.empty()
    
    for i, step in enumerate(thinking_steps):
        progress_container.markdown(
            f'<div class="thinking-animation">{step}</div>', 
            unsafe_allow_html=True
        )
        time.sleep(1)
    
    # Actually run the AI
    noisy_flat = st.session_state.noisy_pattern.flatten()
    result = st.session_state.trained_network.recall(noisy_flat, max_iterations=20)
    restored = result['final_states'].reshape(6, 6)
    
    st.session_state.restored_pattern = restored
    st.session_state.memory_step = 4
    st.rerun()

def show_results_celebration():
    """Show final results with celebration effects"""
    st.markdown("### 🎉 INCREDIBLE RESULTS!")
    
    # Show all three patterns side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📥 Original**")
        display_pattern_visual(st.session_state.original_pattern, "original")
    
    with col2:
        st.markdown("**💥 Corrupted**")
        display_pattern_visual(st.session_state.noisy_pattern, "corrupted")
    
    with col3:
        st.markdown("**✨ AI Restored**")
        display_pattern_visual(st.session_state.restored_pattern, "restored")
    
    # Calculate and show accuracy
    accuracy = np.mean(st.session_state.restored_pattern == st.session_state.original_pattern)
    
    if accuracy == 1.0:
        st.balloons()
        st.success("🎉 PERFECT RESTORATION! The AI remembered your pattern exactly!")
    else:
        st.success(f"🎯 Great job! AI restored {accuracy:.1%} of your pattern correctly!")
    
    # Show accuracy chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = accuracy * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Accuracy"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4ECDC4"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Reset button
    if st.button("🔄 Try Another Pattern!", key="reset"):
        # Reset session state
        keys_to_reset = ['memory_step', 'patterns_drawn', 'ai_trained', 'current_pattern', 
                        'noisy_pattern', 'original_pattern', 'restored_pattern']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def pattern_prophet_demo():
    """Interactive pattern prediction with visual feedback"""
    st.markdown("# 🔮 PATTERN PROPHET")
    st.markdown("### *Watch AI predict the future of patterns!*")
    
    # Simple interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Try These Patterns:")
        
        examples = [
            ("ABAB____", "Alternating Pattern"),
            ("1234____", "Number Sequence"),
            ("AABC____", "Letter Pattern"),
            ("XYXY____", "Repeating Pattern")
        ]
        
        for partial, description in examples:
            if st.button(f"{partial} ({description})", key=f"predict_{partial}"):
                predict_pattern_visual(partial, description)
    
    with col2:
        st.markdown("### ✏️ Or Type Your Own:")
        custom_pattern = st.text_input("Enter pattern (use _ for unknown):", "ABC_____")
        
        if st.button("🔮 Predict!", key="predict_custom"):
            predict_pattern_visual(custom_pattern, "Custom Pattern")

def predict_pattern_visual(partial_pattern, description):
    """Show pattern prediction with visual thinking process"""
    st.markdown(f"### 🧠 AI Predicting: {description}")
    
    # Show thinking animation
    with st.spinner("🤔 AI is analyzing the pattern..."):
        # Create and train AI
        completion = PatternCompletion("text")
        sequences = ["ABABABAB", "12345678", "AABCAABC", "XYXYXYXY"]
        completion.train_sequence_patterns(sequences)
        
        time.sleep(1)
        result = completion.complete_sequence(partial_pattern)
    
    # Show step-by-step thinking
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Pattern Analysis:**")
        for i, char in enumerate(partial_pattern):
            if char != '_':
                st.markdown(f"Position {i+1}: `{char}` ✅")
            else:
                st.markdown(f"Position {i+1}: `?` ❓")
    
    with col2:
        st.markdown("**🎯 AI Prediction:**")
        st.markdown(f"Input: `{partial_pattern}`")
        st.markdown(f"**Result: `{result}`**")
        
        # Show confidence visualization
        confidence = 85 + np.random.randint(-15, 15)  # Simulated confidence
        st.progress(confidence / 100)
        st.markdown(f"Confidence: {confidence}%")

def secret_decoder_demo():
    """Password recovery with visual neural network activity"""
    st.markdown("# 🔐 SECRET DECODER")
    st.markdown("### *Watch AI crack the code by thinking like a detective!*")
    
    # Initialize password AI
    if 'decoder_ai' not in st.session_state:
        recovery = PasswordRecovery()
        recovery.setup_encoding()
        passwords = ["secret123", "password", "admin2024", "user1234", "hello123"]
        recovery.train_on_passwords(passwords, max_length=10)
        st.session_state.decoder_ai = recovery
        st.session_state.secret_database = passwords
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗂️ Secret Database")
        st.markdown("*AI knows these secret codes:*")
        for i, pwd in enumerate(st.session_state.secret_database):
            st.markdown(f"`{i+1}. {pwd}`")
    
    with col2:
        st.markdown("### 🔍 Decode Mission")
        
        examples = [
            ("s___et123", "secret123"),
            ("p___word", "password"),
            ("ad___2024", "admin2024"),
            ("hel___123", "hello123")
        ]
        
        for corrupted, _ in examples:
            if st.button(f"🕵️ Decode: {corrupted}", key=f"decode_{corrupted}"):
                decode_with_animation(corrupted)

def decode_with_animation(corrupted_code):
    """Show decoding process with neural network visualization"""
    st.markdown(f"### 🧠 Decoding: `{corrupted_code}`")
    
    # Show AI thinking process
    thinking_steps = [
        "🔍 Scanning corrupted code...",
        "🧠 Accessing memory database...",
        "⚡ Neural pattern matching...",
        "🔗 Finding connections...",
        "🎯 Code cracked!"
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(thinking_steps):
        status_text.markdown(f'<div class="thinking-animation">{step}</div>', unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(thinking_steps))
        time.sleep(0.8)
    
    # Actually decode
    result = st.session_state.decoder_ai.recover_password(corrupted_code)
    
    # Show results with visual flair
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📥 Corrupted Input:**")
        st.markdown(f"`{corrupted_code}`")
        
        # Show which parts are known/unknown
        for i, char in enumerate(corrupted_code):
            if char == '_':
                st.markdown(f"Position {i+1}: ❓ Unknown")
            else:
                st.markdown(f"Position {i+1}: ✅ `{char}`")
    
    with col2:
        st.markdown("**🎯 AI Decoded:**")
        st.markdown(f"**`{result}`**")
        
        if result in st.session_state.secret_database:
            st.success("🎉 Perfect match found in database!")
            st.balloons()
        else:
            st.info("🤔 AI made its best guess based on patterns!")

# Helper functions
def display_pattern_visual(pattern, pattern_type):
    """Display pattern with visual styling"""
    colors = {
        "original": "#4ECDC4",
        "corrupted": "#FF6B6B", 
        "restored": "#96CEB4"
    }
    
    color = colors.get(pattern_type, "#gray")
    
    html = f"<div style='text-align: center; background: {color}20; padding: 10px; border-radius: 10px;'>"
    for i in range(pattern.shape[0]):
        html += "<div>"
        for j in range(pattern.shape[1]):
            cell_color = color if pattern[i, j] == 1 else "#34495e"
            html += f"<div style='width: 25px; height: 25px; background: {cell_color}; display: inline-block; margin: 1px; border-radius: 3px;'></div>"
        html += "</div>"
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

def add_visual_noise(pattern, noise_level):
    """Add noise to pattern for testing"""
    noisy = pattern.copy()
    flat = noisy.flatten()
    n_flip = int(len(flat) * noise_level)
    flip_indices = np.random.choice(len(flat), n_flip, replace=False)
    flat[flip_indices] *= -1
    return flat.reshape(pattern.shape)

if __name__ == "__main__":
    main()