"""
Visualization Module for Pattern Measures
Author: Brandon Barclay
Date: August 2025

Generate plots and visualizations for the pattern measure framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pattern_measure import PatternMeasure, PatternFamily
from typing import List, Tuple, Optional
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PatternVisualizer:
    """
    Create visualizations for pattern measures and theorems.
    """
    
    def __init__(self, output_dir: str = "../figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_pattern_measure_comparison(self, k_max: int = 30):
        """
        Compare O(k) for different pattern families.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        k_values = np.arange(2, k_max + 1)
        
        # Different pattern families
        families = [
            ("Full exponential: $2^k$", PatternFamily.full_exponential(k_max), 'r-', 2),
            ("Critical: $2^k \\log k$", PatternFamily.exponential_log(k_max), 'b-', 2),
            ("Subcritical: $2^k / (\\log k)^2$", PatternFamily.exponential_polylog(k_max, 2), 'g-', 1.5),
            ("Dimension gap: $2^{0.8k}$", PatternFamily.subexponential(k_max, 0.2), 'm-', 1.5),
        ]
        
        # Plot O(k)
        for name, counts, style, linewidth in families:
            pm = PatternMeasure(counts)
            O_values = [pm.compute_O(k) for k in k_values]
            ax1.semilogy(k_values, O_values, style, label=name, linewidth=linewidth)
        
        ax1.set_xlabel('$k$', fontsize=12)
        ax1.set_ylabel('$O(k)$', fontsize=12)
        ax1.set_title('Pattern Measure $O(k)$ for Different Families', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot entropy gap H_k - k
        for name, counts, style, linewidth in families:
            pm = PatternMeasure(counts)
            gaps = [pm.compute_entropy_gap(k) for k in k_values]
            ax2.plot(k_values, gaps, style, label=name, linewidth=linewidth)
        
        ax2.set_xlabel('$k$', fontsize=12)
        ax2.set_ylabel('$H_k - k$', fontsize=12)
        ax2.set_title('Entropy Gap $H_k - k$', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pattern_measure_comparison.png'), dpi=150)
        plt.show()
        
        print(f"Saved: pattern_measure_comparison.png")
    
    def plot_hierarchy_illustration(self):
        """
        Illustrate the hierarchy of implications with a diagram.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define boxes for each condition
        boxes = [
            (1, 3, 2.5, 1, "Dimension Gap\n$\\limsup H_k/k < 1$", 'lightblue'),
            (4.5, 3, 2.5, 1, "Entropy Gap\n$H_k - k \\to -\\infty$", 'lightgreen'),
            (8, 3, 2.5, 1, "Pattern Decay\n$O(k) \\to 0$", 'lightyellow'),
            (8, 1, 2.5, 1, "Subexp. Growth\n$|P_k| = o(2^k\\log k)$", 'lightcoral'),
        ]
        
        # Draw boxes
        for x, y, w, h, text, color in boxes:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, weight='bold')
        
        # Draw arrows for implications
        arrows = [
            (3.5, 3.5, 1, 0, "⟹"),      # Dimension → Entropy
            (7, 3.5, 1, 0, "⟹"),        # Entropy → Decay
            (9.25, 3, 0, -1, "⟺"),      # Decay ↔ Subexp
        ]
        
        for x, y, dx, dy, label in arrows:
            ax.arrow(x, y, dx*0.8, dy*0.8, head_width=0.15, head_length=0.1, 
                    fc='black', ec='black', linewidth=2)
            if dx != 0:
                ax.text(x + dx/2, y + 0.3, label, ha='center', fontsize=14)
            else:
                ax.text(x + 0.4, y + dy/2, label, ha='center', fontsize=14)
        
        # Add counterexample annotations
        ax.text(5.75, 2.3, "✗", color='red', fontsize=20, weight='bold')
        ax.text(5.75, 1.8, "No reverse", color='red', fontsize=9, ha='center')
        
        ax.text(2.25, 2.3, "✗", color='red', fontsize=20, weight='bold')
        ax.text(2.25, 1.8, "No reverse", color='red', fontsize=9, ha='center')
        
        ax.set_xlim(0, 11.5)
        ax.set_ylim(0.5, 4.5)
        ax.axis('off')
        ax.set_title('Hierarchy of Implications (Strict)', fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hierarchy_diagram.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: hierarchy_diagram.png")
    
    def plot_alpha_exponent_analysis(self):
        """
        Visualize the alpha-exponent framework.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        k_max = 40
        k_values = np.arange(5, k_max + 1)
        
        # Different alpha values
        alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(alphas)))
        
        # Plot 1: O(k) for different alpha
        for alpha, color in zip(alphas, colors):
            counts = PatternFamily.polynomial_exponential(k_max, alpha)
            pm = PatternMeasure(counts)
            O_values = [pm.compute_O(k) for k in k_values]
            ax1.loglog(k_values, O_values, '-', color=color, 
                      label=f'$\\alpha = {alpha}$', linewidth=2)
        
        ax1.set_xlabel('$k$', fontsize=12)
        ax1.set_ylabel('$O(k)$', fontsize=12)
        ax1.set_title('Pattern Measure for Different $\\alpha$-Exponents', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Summability threshold
        alpha_range = np.linspace(0.5, 3, 50)
        partial_sums = []
        
        for alpha in alpha_range:
            counts = PatternFamily.polynomial_exponential(30, alpha)
            pm = PatternMeasure(counts)
            _, partial_sum = pm.check_summability(epsilon=0.01, k_max=30)
            partial_sums.append(partial_sum)
        
        ax2.semilogy(alpha_range, partial_sums, 'b-', linewidth=2)
        ax2.axvline(x=1, color='r', linestyle='--', linewidth=2, label='$\\alpha = 1$ (threshold)')
        ax2.fill_between([0.5, 1], [1e-3, 1e-3], [1e3, 1e3], alpha=0.2, color='red', 
                         label='Divergent')
        ax2.fill_between([1, 3], [1e-3, 1e-3], [1e3, 1e3], alpha=0.2, color='green',
                         label='Convergent')
        
        ax2.set_xlabel('$\\alpha$', fontsize=12)
        ax2.set_ylabel('$\\sum_{k=1}^{30} O(k)^{1.01}$', fontsize=12)
        ax2.set_title('Summability Criterion: $\\alpha > 1$', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.5, 3)
        
        # Plot 3: Alpha estimation accuracy
        true_alphas = np.linspace(0.5, 3, 20)
        estimated_alphas = []
        r_squared_values = []
        
        for alpha_true in true_alphas:
            counts = PatternFamily.polynomial_exponential(50, alpha_true)
            pm = PatternMeasure(counts)
            alpha_est, r2 = pm.compute_alpha_exponent(20, 50)
            estimated_alphas.append(alpha_est)
            r_squared_values.append(r2)
        
        ax3.scatter(true_alphas, estimated_alphas, c=r_squared_values, 
                   cmap='viridis', s=50, alpha=0.7)
        ax3.plot([0.5, 3], [0.5, 3], 'r--', label='Perfect estimation')
        
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('$R^2$', fontsize=10)
        
        ax3.set_xlabel('True $\\alpha$', fontsize=12)
        ax3.set_ylabel('Estimated $\\alpha$', fontsize=12)
        ax3.set_title('$\\alpha$-Exponent Estimation Quality', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trichotomy visualization
        k = np.linspace(2, 30, 100)
        
        # Critical line
        critical = 1 / (np.log(k))**2
        ax4.fill_between(k, 0, critical*0.5, alpha=0.3, color='blue', 
                         label='Subcritical')
        ax4.fill_between(k, critical*0.5, critical*2, alpha=0.3, color='yellow',
                         label='Critical')
        ax4.fill_between(k, critical*2, 1, alpha=0.3, color='red',
                         label='Supercritical')
        
        ax4.semilogy(k, critical, 'k-', linewidth=2, label='$O(k) \\sim 1/(\\log k)^2$')
        
        ax4.set_xlabel('$k$', fontsize=12)
        ax4.set_ylabel('$O(k)$', fontsize=12)
        ax4.set_title('Trichotomy of Pattern Growth', fontsize=13)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(1e-4, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'alpha_exponent_analysis.png'), dpi=150)
        plt.show()
        
        print(f"Saved: alpha_exponent_analysis.png")
    
    def plot_counterexamples(self):
        """
        Visualize the counterexamples showing non-reversibility.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        k_max = 30
        k_values = np.arange(2, k_max + 1)
        
        # Counterexample: |P_k| = 2^k * log(k)
        counts = PatternFamily.exponential_log(k_max)
        pm = PatternMeasure(counts)
        
        # Plot 1: O(k) -> 0
        O_values = [pm.compute_O(k) for k in k_values]
        ax1.semilogy(k_values, O_values, 'b-', linewidth=2)
        ax1.set_xlabel('$k$', fontsize=12)
        ax1.set_ylabel('$O(k)$', fontsize=12)
        ax1.set_title('$O(k) \\to 0$ ✓', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.text(15, 0.1, '$|P_k| = 2^k \\log k$', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # Plot 2: H_k - k NOT -> -infinity
        gaps = [pm.compute_entropy_gap(k) for k in k_values]
        ax2.plot(k_values, gaps, 'r-', linewidth=2)
        ax2.set_xlabel('$k$', fontsize=12)
        ax2.set_ylabel('$H_k - k$', fontsize=12)
        ax2.set_title('$H_k - k \\not\\to -\\infty$ ✗', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 3: H_k/k -> 1 (no dimension gap)
        ratios = [pm.compute_entropy(k) / k for k in k_values if k > 0]
        ax3.plot(k_values, ratios, 'g-', linewidth=2)
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='$y = 1$')
        ax3.set_xlabel('$k$', fontsize=12)
        ax3.set_ylabel('$H_k / k$', fontsize=12)
        ax3.set_title('$\\limsup H_k/k = 1$ (no gap) ✗', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.95, 1.15)
        
        plt.suptitle('Counterexample: Implications Cannot Be Reversed', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'counterexamples.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: counterexamples.png")
    
    def create_all_visualizations(self):
        """
        Generate all visualizations for the paper.
        """
        print("Generating all visualizations...")
        print("-" * 50)
        
        self.plot_pattern_measure_comparison()
        self.plot_hierarchy_illustration()
        self.plot_alpha_exponent_analysis()
        self.plot_counterexamples()
        
        print("-" * 50)
        print(f"All figures saved to {self.output_dir}/")
        print("Visualization complete!")


def main():
    """
    Main visualization routine.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for Pattern Measures paper')
    parser.add_argument('--output', default='../figures', help='Output directory for figures')
    parser.add_argument('--figure', choices=['comparison', 'hierarchy', 'alpha', 'counter', 'all'],
                       default='all', help='Which figure(s) to generate')
    
    args = parser.parse_args()
    
    viz = PatternVisualizer(output_dir=args.output)
    
    if args.figure == 'all':
        viz.create_all_visualizations()
    elif args.figure == 'comparison':
        viz.plot_pattern_measure_comparison()
    elif args.figure == 'hierarchy':
        viz.plot_hierarchy_illustration()
    elif args.figure == 'alpha':
        viz.plot_alpha_exponent_analysis()
    elif args.figure == 'counter':
        viz.plot_counterexamples()


if __name__ == "__main__":
    main()
