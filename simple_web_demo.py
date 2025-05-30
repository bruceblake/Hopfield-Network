#!/usr/bin/env python3
"""
🧠 MIND-READING AI - WEB VERSION 🧠
The simplest, most amazing AI demo ever!
"""

import streamlit as st
import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion
import time

# Page config
st.set_page_config(
    page_title="🧠 Mind-Reading AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #FF6B6B;
    }
    .stButton > button {
        background-color: #4ECDC4;
        color: white;
        font-size: 20px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #45B7B8;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .pattern-cell {
        width: 40px;
        height: 40px;
        margin: 2px;
        display: inline-block;
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .pattern-cell:hover {
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🧠 MIND-READING AI PLAYGROUND 🧠</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Prepare to be amazed! This AI can read patterns like your brain!</p>", 
                unsafe_allow_html=True)
    
    # Sidebar with options
    st.sidebar.markdown("## 🎮 Choose Your Demo")
    demo = st.sidebar.radio(
        "Pick something amazing:",
        ["🎨 Shape Mind Reader", "🔐 Password Wizard", "🔮 Pattern Prophet", "😊 Emoji Genius"]
    )
    
    if demo == "🎨 Shape Mind Reader":
        shape_demo()
    elif demo == "🔐 Password Wizard":
        password_demo()
    elif demo == "🔮 Pattern Prophet":
        pattern_demo()
    else:
        emoji_demo()

def shape_demo():
    st.header("🎨 Shape Mind Reader")
    st.write("### Draw a shape, mess it up, and watch the AI fix it!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Draw Your Shape")
        
        # Initialize pattern in session state
        if 'pattern' not in st.session_state:
            st.session_state.pattern = np.ones((7, 7), dtype=int) * -1
        
        # Drawing interface
        st.write("Click cells to draw (⬛ = filled, ⬜ = empty)")
        
        # Create clickable grid
        for i in range(7):
            cols = st.columns(7)
            for j in range(7):
                key = f"cell_{i}_{j}"
                if cols[j].button("⬛" if st.session_state.pattern[i,j] == 1 else "⬜", key=key):
                    st.session_state.pattern[i,j] *= -1
                    st.rerun()
        
        if st.button("Clear All", key="clear"):
            st.session_state.pattern = np.ones((7, 7), dtype=int) * -1
            st.rerun()
    
    with col2:
        st.subheader("2️⃣ Add Some Noise")
        
        noise_level = st.slider("How much corruption?", 0, 50, 20)
        
        if st.button("🌪️ Corrupt It!", key="corrupt"):
            if 'network' not in st.session_state:
                # Create and train network
                network = HopfieldNetwork(n_neurons=49)
                patterns = generate_preset_patterns()
                patterns.append(st.session_state.pattern.flatten())
                network.train_hebbian(patterns)
                st.session_state.network = network
            
            # Add noise
            flat_pattern = st.session_state.pattern.flatten()
            n_flip = int(49 * noise_level / 100)
            flip_idx = np.random.choice(49, n_flip, replace=False)
            corrupted = flat_pattern.copy()
            corrupted[flip_idx] *= -1
            st.session_state.corrupted = corrupted.reshape(7, 7)
        
        if 'corrupted' in st.session_state:
            st.write("Corrupted version:")
            display_pattern(st.session_state.corrupted)
    
    with col3:
        st.subheader("3️⃣ AI Magic! ✨")
        
        if st.button("🧠 Read My Mind!", key="recall"):
            if 'corrupted' in st.session_state and 'network' in st.session_state:
                with st.spinner("AI is thinking..."):
                    time.sleep(1)
                    recalled = st.session_state.network.recall(
                        st.session_state.corrupted.flatten(), 
                        max_iter=20
                    ).reshape(7, 7)
                    st.session_state.recalled = recalled
        
        if 'recalled' in st.session_state:
            st.write("AI's reconstruction:")
            display_pattern(st.session_state.recalled)
            
            if np.array_equal(st.session_state.recalled, st.session_state.pattern):
                st.balloons()
                st.success("🎉 PERFECT! The AI read your mind!")
            else:
                st.info("🤔 AI found a similar pattern in its memory!")

def display_pattern(pattern):
    """Display pattern as emoji grid."""
    html = "<div style='text-align: center;'>"
    for row in pattern:
        html += "<div>"
        for val in row:
            color = "black" if val == 1 else "white"
            html += f"<div class='pattern-cell' style='background-color: {color};'></div>"
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def generate_preset_patterns():
    """Generate some preset patterns for the network."""
    return [
        # Cross
        np.array([
            [-1,-1,-1,1,-1,-1,-1],
            [-1,-1,-1,1,-1,-1,-1],
            [-1,-1,-1,1,-1,-1,-1],
            [1,1,1,1,1,1,1],
            [-1,-1,-1,1,-1,-1,-1],
            [-1,-1,-1,1,-1,-1,-1],
            [-1,-1,-1,1,-1,-1,-1]
        ]).flatten(),
        # Square
        np.array([
            [1,1,1,1,1,1,1],
            [1,-1,-1,-1,-1,-1,1],
            [1,-1,-1,-1,-1,-1,1],
            [1,-1,-1,-1,-1,-1,1],
            [1,-1,-1,-1,-1,-1,1],
            [1,-1,-1,-1,-1,-1,1],
            [1,1,1,1,1,1,1]
        ]).flatten()
    ]

def password_demo():
    st.header("🔐 Password Recovery Wizard")
    st.write("### Forgot your password? The AI remembers!")
    
    # Initialize password recovery
    if 'pwd_recovery' not in st.session_state:
        recovery = PasswordRecovery(password_length=8)
        passwords = [
            "mydog123", "pizza888", "love2024", 
            "star*123", "cool_cat", "happy999",
            "blue#sky", "code4fun"
        ]
        recovery.train_passwords(passwords)
        st.session_state.pwd_recovery = recovery
        st.session_state.pwd_list = passwords
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 AI knows these passwords:")
        for pwd in st.session_state.pwd_list:
            st.code(pwd)
    
    with col2:
        st.subheader("🤔 What do you remember?")
        
        st.write("Use _ for forgotten characters")
        corrupted = st.text_input("Enter partial password:", "m_dog___")
        
        if st.button("🧠 Recover Password!", key="recover"):
            with st.spinner("AI searching its memory..."):
                time.sleep(1)
                recovered = st.session_state.pwd_recovery.recover_password(corrupted)
                
            st.success(f"✨ AI says: **{recovered}**")
            
            if recovered in st.session_state.pwd_list:
                st.balloons()
                st.write("🎉 That's a valid password from the AI's memory!")

def pattern_demo():
    st.header("🔮 Pattern Prophet")
    st.write("### The AI can predict patterns like magic!")
    
    # Initialize pattern completion
    if 'pattern_comp' not in st.session_state:
        completion = PatternCompletion(
            sequence_length=8,
            vocabulary=['A', 'B', 'C', 'D', 'E', 'X', 'Y', 'Z']
        )
        sequences = [
            "ABCDABCD",
            "ABCDEFGH",
            "AABBCCDD",
            "ABABABAB",
            "XYZXYZXY"
        ]
        completion.train_sequences(sequences)
        st.session_state.pattern_comp = completion
        st.session_state.sequences = sequences
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 AI learned these patterns:")
        for seq in st.session_state.sequences:
            st.code(seq)
    
    with col2:
        st.subheader("🎯 Complete the pattern!")
        
        partial = st.text_input("Enter partial pattern (use _ for missing):", "ABCD____")
        
        if st.button("🔮 Predict!", key="predict"):
            with st.spinner("AI is predicting the future..."):
                time.sleep(1)
                completed = st.session_state.pattern_comp.complete_sequence(partial)
            
            st.success(f"✨ AI predicts: **{completed}**")
            st.write("🎯 Does that make sense? The AI found the pattern!")

def emoji_demo():
    st.header("😊 Emoji Genius")
    st.write("### The AI speaks emoji fluently!")
    
    # Initialize emoji completion
    if 'emoji_comp' not in st.session_state:
        completion = PatternCompletion(
            sequence_length=6,
            vocabulary=['😊', '😎', '🎉', '❤️', '🌟', '🚀', '🎨', '🎵', '🔥', '💎']
        )
        sequences = [
            "😊😎🎉❤️🌟🚀",
            "🎨🎵🎨🎵🎨🎵",
            "❤️❤️😊😊🎉🎉",
            "🔥💎🔥💎🔥💎",
            "🚀🌟🚀🌟🚀🌟"
        ]
        completion.train_sequences(sequences)
        st.session_state.emoji_comp = completion
        st.session_state.emoji_seqs = sequences
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 AI learned these emoji patterns:")
        for seq in st.session_state.emoji_seqs:
            st.write(seq)
    
    with col2:
        st.subheader("🎯 What comes next?")
        
        # Emoji selector
        st.write("Build your pattern (use the last 2 spots for AI to complete):")
        
        cols = st.columns(6)
        pattern = []
        for i in range(4):
            emoji = cols[i].selectbox(
                f"Pos {i+1}", 
                ['😊', '😎', '🎉', '❤️', '🌟', '🚀', '🎨', '🎵', '🔥', '💎'],
                key=f"emoji_{i}"
            )
            pattern.append(emoji)
        
        partial = "".join(pattern) + "__"
        st.write(f"Your pattern: {partial}")
        
        if st.button("🧠 Complete it!", key="emoji_complete"):
            with st.spinner("AI thinking in emojis..."):
                time.sleep(1)
                completed = st.session_state.emoji_comp.complete_sequence(partial)
            
            st.success(f"✨ AI says: **{completed}**")
            st.write("🎉 The AI learned the emoji language!")

if __name__ == "__main__":
    main()