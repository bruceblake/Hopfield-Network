#!/usr/bin/env python3
"""
🧠 MIND-READING NEURAL NETWORK DEMO 🧠
This AI can read your mind and fix your memories!
"""

import numpy as np
from hopfield_toolkit import HopfieldNetwork
from applications import PasswordRecovery, PatternCompletion
import time
import random

def print_slow(text, delay=0.03):
    """Print text with a typewriter effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def print_pattern(pattern, size=5):
    """Display pattern as emoji art."""
    pattern = pattern.reshape(size, size)
    for row in pattern:
        for val in row:
            print("⬛" if val == 1 else "⬜", end=" ")
        print()

def mind_reading_demo():
    """Demo 1: The AI reads your mind!"""
    print("\n" + "="*50)
    print("🧠 DEMO 1: MIND-READING AI 🧠")
    print("="*50)
    
    print_slow("\nI'm going to show you some shapes and then read your mind!")
    print_slow("Watch carefully...\n")
    
    time.sleep(1)
    
    # Create network
    network = HopfieldNetwork(n_neurons=25, name="MindReader")
    
    # Define simple patterns
    patterns = {
        "CROSS": np.array([
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1,
            1, 1, 1, 1, 1,
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1
        ]),
        "SQUARE": np.array([
            1, 1, 1, 1, 1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            1, 1, 1, 1, 1
        ]),
        "DIAMOND": np.array([
            -1, -1, 1, -1, -1,
            -1, 1, -1, 1, -1,
            1, -1, -1, -1, 1,
            -1, 1, -1, 1, -1,
            -1, -1, 1, -1, -1
        ])
    }
    
    # Show patterns
    for name, pattern in patterns.items():
        print(f"\n📌 Remember this - {name}:")
        print_pattern(pattern)
        time.sleep(1.5)
    
    # Train the AI
    print_slow("\n🧠 The AI is memorizing these shapes...")
    network.train_hebbian(list(patterns.values()))
    time.sleep(1)
    print_slow("✅ Done! The AI has learned your patterns!\n")
    
    # Now mess up a pattern
    print_slow("Now I'll show you a corrupted shape...")
    print_slow("Can you guess which one it's supposed to be?\n")
    
    # Corrupt the square
    corrupted = patterns["SQUARE"].copy()
    # Add 40% noise
    noise_positions = random.sample(range(25), 10)
    for pos in noise_positions:
        corrupted[pos] *= -1
    
    print("🤔 What shape is this supposed to be?")
    print_pattern(corrupted)
    
    input("\nPress Enter when you've made your guess...")
    
    # AI recalls
    print_slow("\n🧠 Let the AI read your mind...")
    time.sleep(1)
    
    recalled = network.recall(corrupted, max_iter=10)
    
    print("\n✨ The AI says it's a...")
    time.sleep(1)
    print_pattern(recalled)
    
    if np.array_equal(recalled, patterns["SQUARE"]):
        print_slow("\n🎉 SQUARE! The AI read your mind correctly!")
    
    print_slow("\nThe AI can restore corrupted memories perfectly! 🤯")

def password_magic_demo():
    """Demo 2: Password recovery magic!"""
    print("\n\n" + "="*50)
    print("🔐 DEMO 2: PASSWORD RECOVERY MAGIC 🔐")
    print("="*50)
    
    print_slow("\nEver forgot parts of your password? This AI can help!")
    print_slow("Let me show you some magic...\n")
    
    recovery = PasswordRecovery(password_length=8)
    
    # Secret passwords
    passwords = [
        "mydog123",
        "pizza888", 
        "love2024",
        "star*123",
        "cool_cat"
    ]
    
    print("📝 I'm teaching the AI these passwords:")
    for pwd in passwords:
        print(f"   • {pwd}")
        time.sleep(0.5)
    
    recovery.train_passwords(passwords)
    print_slow("\n✅ AI has memorized the passwords!\n")
    
    # Demo corrupted passwords
    demos = [
        ("m_dog___", "mydog123"),
        ("p___a888", "pizza888"),
        ("__ve2024", "love2024")
    ]
    
    for corrupted, original in demos:
        print(f"\n🤔 You remember: {corrupted}")
        print_slow("   (where _ means forgotten character)")
        
        print_slow("\n🧠 AI is thinking...")
        recovered = recovery.recover_password(corrupted)
        time.sleep(1)
        
        print(f"\n✨ AI recovered: {recovered}")
        if recovered == original:
            print_slow("   ✅ CORRECT! The AI saved the day!")
        time.sleep(1)

def autocomplete_demo():
    """Demo 3: Superhuman autocomplete!"""
    print("\n\n" + "="*50)
    print("🎯 DEMO 3: SUPERHUMAN AUTOCOMPLETE 🎯")
    print("="*50)
    
    print_slow("\nThis AI can complete patterns better than humans!")
    print_slow("Watch this...\n")
    
    # Number sequences
    completion = PatternCompletion(
        sequence_length=8,
        vocabulary=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    )
    
    sequences = [
        "12345678",
        "13579135",
        "24681357",
        "11223344",
        "98765432"
    ]
    
    print("📚 Teaching the AI these number patterns:")
    for seq in sequences:
        print(f"   • {seq}")
        time.sleep(0.5)
    
    completion.train_sequences(sequences)
    print_slow("\n✅ AI has learned the patterns!\n")
    
    # Test sequences
    tests = [
        "1234____",
        "1357____",
        "2468____",
        "1122____"
    ]
    
    for partial in tests:
        print(f"\n🤔 Complete this: {partial}")
        print_slow("   (where _ means missing)")
        
        print_slow("\n🧠 AI is predicting...")
        completed = completion.complete_sequence(partial)
        time.sleep(1)
        
        print(f"\n✨ AI says: {completed}")
        print_slow("   🎯 Makes sense, right?")
        time.sleep(1)

def emoji_demo():
    """Demo 4: Emoji mind reading!"""
    print("\n\n" + "="*50)
    print("😊 DEMO 4: EMOJI MIND READER 😊")
    print("="*50)
    
    print_slow("\nThe AI can even work with emojis!")
    print_slow("This is really cool...\n")
    
    # Emoji patterns
    completion = PatternCompletion(
        sequence_length=5,
        vocabulary=['😊', '😎', '🎉', '❤️', '🌟', '🚀', '🎨', '🎵']
    )
    
    sequences = [
        "😊😎🎉❤️🌟",
        "🚀🌟🚀🌟🚀",
        "🎨🎵🎨🎵🎨",
        "❤️❤️😊😊🎉"
    ]
    
    print("🎯 Teaching these emoji patterns:")
    for seq in sequences:
        print(f"   • {seq}")
        time.sleep(0.8)
    
    completion.train_sequences(sequences)
    print_slow("\n✅ AI learned the emoji language!\n")
    
    # Interactive part
    print_slow("Now YOU try!")
    print("Complete this pattern: 😊😎🎉__")
    print("\nWhat comes next? Type 1, 2, or 3:")
    print("1) ❤️🌟")
    print("2) 🚀🎨") 
    print("3) 😊😎")
    
    user_choice = input("\nYour guess: ")
    
    print_slow("\n🧠 Let's see what the AI thinks...")
    completed = completion.complete_sequence("😊😎🎉__")
    time.sleep(1)
    
    print(f"\n✨ AI says: {completed}")
    if "❤️🌟" in completed:
        print_slow("🎉 The AI agrees with option 1! It learned the pattern!")

def interactive_finale():
    """Interactive finale!"""
    print("\n\n" + "="*60)
    print("🌟 WANT TO TRY IT YOURSELF? 🌟")
    print("="*60)
    
    print_slow("\nThis AI technology is called a Hopfield Network!")
    print_slow("It's like a brain that can:")
    print_slow("  ✨ Restore corrupted memories")
    print_slow("  ✨ Complete missing information") 
    print_slow("  ✨ Recognize patterns")
    print_slow("  ✨ Fix errors automatically")
    
    print("\n🚀 Real-world uses:")
    print("  • Password recovery systems")
    print("  • Image restoration (fix old photos!)")
    print("  • Autocomplete & predictive text")
    print("  • Pattern recognition")
    print("  • Memory enhancement apps")
    
    print_slow("\n💡 The coolest part? It works like your brain!")
    print_slow("   Just like you can recognize a friend even if")
    print_slow("   they're wearing sunglasses, this AI can recognize")
    print_slow("   patterns even when they're partially hidden!")

def main():
    """Run all demos."""
    print("\n" + "🧠"*25)
    print("WELCOME TO THE MIND-READING AI DEMO!")
    print("🧠"*25)
    
    print_slow("\nPrepare to be amazed by artificial intelligence...")
    input("\nPress Enter to begin the magic show! ")
    
    # Run demos
    mind_reading_demo()
    input("\n\n👉 Press Enter for the next demo...")
    
    password_magic_demo()
    input("\n\n👉 Press Enter for the next demo...")
    
    autocomplete_demo()
    input("\n\n👉 Press Enter for the final demo...")
    
    emoji_demo()
    
    # Finale
    interactive_finale()
    
    print("\n\n🎉 Thanks for watching! 🎉")
    print("Want to learn more? Visit the web interface!")
    print("\n" + "🧠"*25 + "\n")

if __name__ == "__main__":
    main()