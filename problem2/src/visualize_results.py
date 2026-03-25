"""
Visualization Script for Results
=================================

Creates plots and visualizations for model comparison.

Usage:
    python visualize_results.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(results_file='detailed_evaluation.json'):
    """Create comparison plots"""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    models = list(results.keys())
    
    # Extract metrics
    novelty = [results[m]['novelty_rate'] for m in models]
    diversity = [results[m]['diversity'] for m in models]
    realism = [results[m]['realism']['realism_rate'] for m in models]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RNN Model Comparison - Name Generation', fontsize=16, fontweight='bold')
    
    # 1. Bar chart - Novelty Rate
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, novelty, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Novelty Rate (%)', fontweight='bold')
    ax1.set_title('Novelty Rate (Higher is Better)')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Bar chart - Diversity
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, diversity, color=['#9b59b6', '#f39c12', '#1abc9c'])
    ax2.set_ylabel('Diversity (%)', fontweight='bold')
    ax2.set_title('Diversity (Higher is Better)')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. Bar chart - Realism
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, realism, color=['#e67e22', '#16a085', '#c0392b'])
    ax3.set_ylabel('Realism Rate (%)', fontweight='bold')
    ax3.set_title('Realism Rate (Higher is Better)')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 4. Radar chart - Overall comparison
    ax4 = axes[1, 1]
    ax4.remove()  # Remove to create polar plot
    ax4 = fig.add_subplot(224, projection='polar')
    
    # Prepare data for radar chart
    categories = ['Novelty', 'Diversity', 'Realism']
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, model in enumerate(models):
        values = [
            results[model]['novelty_rate'],
            results[model]['diversity'],
            results[model]['realism']['realism_rate']
        ]
        values += values[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax4.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 100)
    ax4.set_title('Overall Performance', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot: model_comparison.png")
    
    plt.show()


def plot_name_length_distribution(results_file='detailed_evaluation.json'):
    """Plot name length distributions"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Name Length Distribution by Model', fontsize=14, fontweight='bold')
    
    for idx, (model, data) in enumerate(results.items()):
        # Get sample names
        samples = data.get('samples', [])
        lengths = [len(name) for name in samples]
        
        ax = axes[idx]
        ax.hist(lengths, bins=range(3, 16), color=f'C{idx}', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Name Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title(model)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        avg_len = np.mean(lengths)
        ax.axvline(avg_len, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {avg_len:.1f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('length_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved length distribution: length_distribution.png")
    
    plt.show()


def create_summary_table(results_file='detailed_evaluation.json'):
    """Create a summary table"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY TABLE")
    print("="*80 + "\n")
    
    # Header
    print(f"{'Model':25s} {'Novelty':>12s} {'Diversity':>12s} {'Realism':>12s} {'Avg Length':>12s}")
    print("-" * 80)
    
    # Data rows
    for model, data in results.items():
        print(f"{model:25s} "
              f"{data['novelty_rate']:>11.2f}% "
              f"{data['diversity']:>11.2f}% "
              f"{data['realism']['realism_rate']:>11.2f}% "
              f"{data['structure']['avg_length']:>11.2f}")
    
    print("\n" + "="*80)
    
    # Best performers
    print("\n🏆 BEST PERFORMERS:")
    
    best_novelty = max(results.items(), key=lambda x: x[1]['novelty_rate'])
    print(f"   Highest Novelty:  {best_novelty[0]} ({best_novelty[1]['novelty_rate']:.2f}%)")
    
    best_diversity = max(results.items(), key=lambda x: x[1]['diversity'])
    print(f"   Highest Diversity: {best_diversity[0]} ({best_diversity[1]['diversity']:.2f}%)")
    
    best_realism = max(results.items(), key=lambda x: x[1]['realism']['realism_rate'])
    print(f"   Highest Realism:  {best_realism[0]} ({best_realism[1]['realism']['realism_rate']:.2f}%)")
    
    print("\n" + "="*80)


def main():
    """Main visualization function"""
    import os
    
    if not os.path.exists('detailed_evaluation.json'):
        print("Error: detailed_evaluation.json not found!")
        print("   Please run evaluate_and_report.py first.")
        return
    
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Create plots
    try:
        plot_model_comparison()
        plot_name_length_distribution()
        create_summary_table()
        
        print("\nAll visualizations created successfully!")
        print("\nGenerated files:")
        print("  • model_comparison.png")
        print("  • length_distribution.png")
        
    except Exception as e:
        print(f"\nError creating visualizations: {str(e)}")
        print("   Make sure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    main()
