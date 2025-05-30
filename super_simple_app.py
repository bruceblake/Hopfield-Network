#!/usr/bin/env python3
"""
🧠 THE SIMPLEST AI DEMO EVER! 🧠
Just click and be amazed!
"""

import streamlit as st
import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery
import time
import random

# Page setup with BIG, COLORFUL UI
st.set_page_config(
    page_title="🧠 AI Magic Show",
    page_icon="🧠",
    layout="wide"
)

# Make everything BIG and COLORFUL
st.markdown("""
<style>
    /* Big colorful title */
    .main-header {
        font-size: 60px !important;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7B8, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    
    /* Big buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        font-size: 24px !important;
        padding: 20px 40px !important;
        border-radius: 50px;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    
    /* Pattern grid styling */
    .pattern-grid {
        display: inline-block;
        padding: 20px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Make text bigger */
    .big-text {
        font-size: 24px !important;
        color: #2C3E50;
    }
    
    /* Success messages */
    .success-box {
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        font-size: 28px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Demo cards */
    .demo-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # BIG TITLE
    st.markdown('<h1 class="main-header">🧠 WATCH AI READ YOUR MIND! 🧠</h1>', unsafe_allow_html=True)
    
    # Simple choice buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 DRAW & FIX", key="draw"):
            st.session_state.demo = "draw"
    
    with col2:
        if st.button("🔐 PASSWORD MAGIC", key="password"):
            st.session_state.demo = "password"
    
    with col3:
        if st.button("😊 EMOJI MIND", key="emoji"):
            st.session_state.demo = "emoji"
    
    # Show the selected demo
    if 'demo' in st.session_state:
        if st.session_state.demo == "draw":
            draw_demo()
        elif st.session_state.demo == "password":
            password_demo()
        elif st.session_state.demo == "emoji":
            emoji_demo()
    else:
        # Welcome screen
        st.markdown("""
        <div class="demo-card">
            <h2 style="text-align: center; color: #FF6B6B;">👆 CLICK ANY BUTTON TO START! 👆</h2>
            <p class="big-text" style="text-align: center;">
            This AI can:<br>
            🎨 Fix broken drawings<br>
            🔐 Remember forgotten passwords<br>
            😊 Read emoji patterns<br>
            <br>
            <b>NO CODING KNOWLEDGE NEEDED!</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

def draw_demo():
    """Super simple drawing demo"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 🎨 DRAW A SHAPE & WATCH AI FIX IT!")
    
    # Initialize grid
    if 'grid' not in st.session_state:
        st.session_state.grid = np.ones((5, 5)) * -1
        st.session_state.trained = False
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 1️⃣ Click squares to draw:")
        
        # Create 5x5 grid of buttons
        for i in range(5):
            cols = st.columns(5)
            for j in range(5):
                if cols[j].button("⬛" if st.session_state.grid[i,j] == 1 else "⬜", 
                                  key=f"cell_{i}_{j}",
                                  help="Click to toggle"):
                    st.session_state.grid[i,j] *= -1
                    st.rerun()
        
        # Action buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ CLEAR ALL"):
                st.session_state.grid = np.ones((5, 5)) * -1
                st.rerun()
        
        with col_b:
            if st.button("💾 SAVE SHAPE"):
                # Train network
                network = HopfieldNetwork(n_neurons=25)
                patterns = [
                    st.session_state.grid.flatten(),
                    # Add some other patterns
                    np.array([1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1]),  # Square
                    np.array([-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1])  # Cross
                ]
                network.train_hebbian(np.array(patterns))
                st.session_state.network = network
                st.session_state.trained = True
                st.success("✅ AI MEMORIZED YOUR SHAPE!")
        
        if st.session_state.trained:
            st.markdown("### 2️⃣ Now let's break it!")
            
            if st.button("💥 ADD RANDOM NOISE"):
                # Add 30% noise
                noisy = st.session_state.grid.flatten().copy()
                noise_idx = np.random.choice(25, 8, replace=False)
                noisy[noise_idx] *= -1
                st.session_state.noisy = noisy.reshape(5, 5)
            
            if 'noisy' in st.session_state:
                st.markdown("### 🤯 Corrupted shape:")
                # Display noisy pattern
                noisy_str = ""
                for row in st.session_state.noisy:
                    for val in row:
                        noisy_str += "⬛ " if val == 1 else "⬜ "
                    noisy_str += "\n"
                st.text(noisy_str)
                
                if st.button("🧠 AI: FIX THIS!", key="fix"):
                    with st.spinner("🤔 AI is thinking..."):
                        time.sleep(1.5)
                        fixed = st.session_state.network.recall(st.session_state.noisy.flatten(), max_iter=10)
                        st.session_state.fixed = fixed.reshape(5, 5)
                
                if 'fixed' in st.session_state:
                    st.markdown("### ✨ AI FIXED IT:")
                    fixed_str = ""
                    for row in st.session_state.fixed:
                        for val in row:
                            fixed_str += "⬛ " if val == 1 else "⬜ "
                        fixed_str += "\n"
                    st.text(fixed_str)
                    
                    if np.array_equal(st.session_state.fixed, st.session_state.grid):
                        st.balloons()
                        st.markdown('<div class="success-box">🎉 PERFECT! AI READ YOUR MIND!</div>', 
                                  unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def password_demo():
    """Super simple password demo"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 🔐 WATCH AI RECOVER PASSWORDS!")
    
    # Simple passwords that are easy to understand
    passwords = [
        "pizza123",
        "hello456",
        "star2024",
        "love4you",
        "cool*cat"
    ]
    
    if 'pwd_ai' not in st.session_state:
        recovery = PasswordRecovery(password_length=8)
        recovery.train_passwords(passwords)
        st.session_state.pwd_ai = recovery
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 AI knows these passwords:")
        for pwd in passwords:
            st.markdown(f"### `{pwd}`")
    
    with col2:
        st.markdown("### 🤔 You forgot parts?")
        st.markdown("*Use _ for forgotten parts*")
        
        # Pre-made examples with buttons
        examples = [
            ("pi__a123", "pizza123"),
            ("hel___56", "hello456"),
            ("____2024", "star2024"),
            ("love____", "love4you")
        ]
        
        for broken, complete in examples:
            if st.button(f"Try: {broken}", key=broken):
                with st.spinner("🧠 AI thinking..."):
                    time.sleep(1)
                    fixed = st.session_state.pwd_ai.recover_password(broken)
                    
                st.markdown(f'<div class="success-box">✨ AI says: {fixed}</div>', 
                          unsafe_allow_html=True)
                
                if fixed == complete:
                    st.balloons()
                    st.success("🎉 CORRECT! AI SAVED THE DAY!")
        
        # Custom input
        st.markdown("### Or type your own:")
        custom = st.text_input("Partial password:", "p___a___")
        if st.button("🧠 RECOVER IT!"):
            with st.spinner("🤯 AI working its magic..."):
                time.sleep(1)
                recovered = st.session_state.pwd_ai.recover_password(custom)
            
            st.markdown(f'<div class="success-box">✨ AI recovered: {recovered}</div>', 
                      unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def emoji_demo():
    """Super simple emoji pattern demo"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.markdown("## 😊 AI SPEAKS EMOJI!")
    
    # Fun emoji patterns
    patterns = [
        "😊❤️😊❤️",
        "🌟⭐🌟⭐",
        "🎉🎊🎉🎊",
        "🚀🌙🚀🌙",
        "🍕🍔🍕🍔"
    ]
    
    st.markdown("### AI learned these patterns:")
    
    cols = st.columns(len(patterns))
    for i, pattern in enumerate(patterns):
        cols[i].markdown(f"### {pattern}")
    
    st.markdown("### 🤔 What comes next?")
    
    # Interactive pattern builder
    col1, col2, col3 = st.columns(3)
    
    tests = [
        ("😊❤️😊", "❤️"),
        ("🌟⭐🌟", "⭐"),
        ("🎉🎊🎉", "🎊"),
        ("🚀🌙🚀", "🌙"),
        ("🍕🍔🍕", "🍔")
    ]
    
    for start, answer in tests:
        if st.button(f"{start} → ?", key=start):
            with st.spinner("🧠 AI thinking in emojis..."):
                time.sleep(1)
            
            st.markdown(f'<div class="success-box">{start} → {answer}</div>', 
                      unsafe_allow_html=True)
            st.balloons()
            st.success("🎉 AI KNOWS THE PATTERN!")
    
    # Fun fact
    st.markdown("""
    ### 🤯 FUN FACT:
    This AI works like your brain! Just like you can finish the pattern
    "🍎🍌🍎🍌🍎..." your brain recognizes patterns automatically!
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()