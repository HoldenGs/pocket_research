#!/usr/bin/env python3
"""
Visualize the effect of weighted sampling on steering data distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

def visualize_weighted_sampling():
    """Create visualizations showing weighted sampling effects"""
    
    # Simulate realistic steering data distribution
    np.random.seed(42)
    
    # Create sample steering data (mimicking real distribution)
    straight_samples = np.random.normal(0, 0.05, 1000)  # Mostly straight
    light_turns = np.concatenate([
        np.random.normal(-0.2, 0.05, 50),  # Left light turns
        np.random.normal(0.2, 0.05, 50)    # Right light turns
    ])
    sharp_turns = np.concatenate([
        np.random.normal(-0.5, 0.1, 25),   # Left sharp turns  
        np.random.normal(0.5, 0.1, 25)    # Right sharp turns
    ])
    
    all_steering = np.concatenate([straight_samples, light_turns, sharp_turns])
    np.random.shuffle(all_steering)  # Mix them up
    
    # Apply same categorization as in fix_steering_problem.py
    abs_steering = np.abs(all_steering)
    straight = abs_steering < 0.1
    light_turn = (abs_steering >= 0.1) & (abs_steering < 0.3)
    sharp_turn = abs_steering >= 0.3
    
    # Create weights (same as in your code)
    weights = np.ones(len(all_steering))
    weights[straight] = 1.0
    weights[light_turn] = 3.0  
    weights[sharp_turn] = 10.0
    
    print("📊 Original Distribution:")
    print(f"  Straight: {np.sum(straight)} ({100*np.sum(straight)/len(all_steering):.1f}%)")
    print(f"  Light turns: {np.sum(light_turn)} ({100*np.sum(light_turn)/len(all_steering):.1f}%)")
    print(f"  Sharp turns: {np.sum(sharp_turn)} ({100*np.sum(sharp_turn)/len(all_steering):.1f}%)")
    
    # Simulate weighted sampling
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    sampled_indices = list(sampler)[:5000]  # Sample 5000 examples
    sampled_steering = all_steering[sampled_indices]
    
    # Categorize sampled data
    sampled_abs = np.abs(sampled_steering)
    sampled_straight = sampled_abs < 0.1
    sampled_light = (sampled_abs >= 0.1) & (sampled_abs < 0.3)
    sampled_sharp = sampled_abs >= 0.3
    
    print("\n🎲 After Weighted Sampling:")
    print(f"  Straight: {np.sum(sampled_straight)} ({100*np.sum(sampled_straight)/len(sampled_steering):.1f}%)")
    print(f"  Light turns: {np.sum(sampled_light)} ({100*np.sum(sampled_light)/len(sampled_steering):.1f}%)")
    print(f"  Sharp turns: {np.sum(sampled_sharp)} ({100*np.sum(sampled_sharp)/len(sampled_steering):.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram comparison
    bins = np.linspace(-0.8, 0.8, 50)
    axes[0,0].hist(all_steering, bins=bins, alpha=0.7, label='Original', color='lightblue')
    axes[0,0].hist(sampled_steering, bins=bins, alpha=0.7, label='Weighted Sample', color='orange')
    axes[0,0].set_xlabel('Steering Angle')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Steering Distribution: Before vs After Weighted Sampling')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Category distribution pie charts
    original_counts = [np.sum(straight), np.sum(light_turn), np.sum(sharp_turn)]
    sampled_counts = [np.sum(sampled_straight), np.sum(sampled_light), np.sum(sampled_sharp)]
    categories = ['Straight\n(|θ| < 0.1)', 'Light Turn\n(0.1 ≤ |θ| < 0.3)', 'Sharp Turn\n(|θ| ≥ 0.3)']
    colors = ['lightgreen', 'gold', 'salmon']
    
    axes[0,1].pie(original_counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Original Distribution')
    
    axes[1,0].pie(sampled_counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1,0].set_title('After Weighted Sampling')
    
    # 3. Weight visualization
    weight_categories = ['Straight\n(weight=1.0)', 'Light Turn\n(weight=3.0)', 'Sharp Turn\n(weight=10.0)']
    weight_values = [1.0, 3.0, 10.0]
    
    bars = axes[1,1].bar(weight_categories, weight_values, color=colors)
    axes[1,1].set_ylabel('Sampling Weight')
    axes[1,1].set_title('Sampling Weights by Category')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, weight_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                      f'{value}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('weighted_sampling_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create sampling probability visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate actual probabilities
    total_weight = np.sum(weights)
    prob_straight = weights[straight][0] / total_weight
    prob_light = weights[light_turn][0] / total_weight if np.any(light_turn) else 0
    prob_sharp = weights[sharp_turn][0] / total_weight if np.any(sharp_turn) else 0
    
    # Show individual sample probabilities
    sample_types = ['Straight Sample', 'Light Turn Sample', 'Sharp Turn Sample']
    probabilities = [prob_straight, prob_light, prob_sharp]
    multipliers = [1, 3, 10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Probability per sample
    bars1 = ax1.bar(sample_types, probabilities, color=colors)
    ax1.set_ylabel('Probability of Selection')
    ax1.set_title('Individual Sample Selection Probability')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, prob in zip(bars1, probabilities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + prob*0.05, 
                f'{prob:.1e}', ha='center', va='bottom', fontweight='bold')
    
    # Relative multiplier effect
    bars2 = ax2.bar(sample_types, multipliers, color=colors)
    ax2.set_ylabel('Relative Likelihood (vs Straight)')
    ax2.set_title('Sampling Likelihood Multiplier')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mult in zip(bars2, multipliers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{mult}x', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sampling_probabilities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualizations saved:")
    print(f"  - weighted_sampling_visualization.png")
    print(f"  - sampling_probabilities.png")

if __name__ == "__main__":
    visualize_weighted_sampling() 