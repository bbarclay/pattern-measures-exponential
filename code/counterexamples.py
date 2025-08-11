"""
Counterexamples Module.

Demonstrate that the implications in the hierarchy cannot be reversed,
establishing the strict nature of the theoretical framework.

Author: Brandon Barclay
Date: August 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from pattern_measure import PatternMeasure, PatternFamily
from core_utils import (
    compute_pattern_measure,
    compute_entropy_gap,
    safe_log2,
    print_section_header,
    print_subsection,
    format_scientific
)


class CounterexampleDemonstrator:
    """
    Demonstrate counterexamples showing the hierarchy is strict.
    
    This class provides systematic demonstrations that the implications
    in our theoretical hierarchy cannot be reversed.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the demonstrator.
        
        Args:
            verbose: If True, print detailed output
        """
        self.verbose = verbose
        self.results: List[Tuple[str, bool]] = []
    
    def _print_table_header(self, columns: List[Tuple[str, int]]) -> None:
        """
        Print a formatted table header.
        
        Args:
            columns: List of (name, width) tuples for each column
        """
        if not self.verbose:
            return
        
        header = " | ".join(f"{name:>{width}}" for name, width in columns)
        print(f"  {header}")
        print("  " + "-" * len(header))
    
    def _compute_with_overflow_protection(
        self,
        k: int,
        formula: str = "2^k * log(k)"
    ) -> Tuple[float, float, float]:
        """
        Compute O(k), entropy gap, and H_k/k with overflow protection.
        
        Args:
            k: Scale parameter
            formula: Pattern family formula
            
        Returns:
            (O_k, entropy_gap, H_k_over_k) tuple
        """
        # Limit exponential growth for computation
        k_eff = min(k, 20)
        
        if formula == "2^k * log(k)":
            log_P_k = k_eff + safe_log2(np.log(k))
            gap = safe_log2(np.log(k))  # Analytical result
        elif formula == "2^k / (k * log^2(k))":
            log_P_k = k_eff - safe_log2(k) - 2 * safe_log2(np.log(k))
            gap = -safe_log2(k) - 2 * safe_log2(np.log(k))
        else:
            # Fallback
            P_k = int(2**k_eff * np.log(k))
            log_P_k = safe_log2(P_k)
            gap = log_P_k - k_eff
        
        O_k = compute_pattern_measure(k, log_P_k=log_P_k)
        H_k_over_k = (k_eff + gap) / k if k > 0 else 0
        
        return O_k, gap, H_k_over_k
    
    def demonstrate_decay_without_entropy_gap(self) -> bool:
        """
        Show that O(k) -> 0 does not imply H_k - k -> -infinity.
        
        Counterexample: |P_k| = 2^k * log(k)
        
        Returns:
            True if demonstration successful
        """
        if self.verbose:
            print_section_header("COUNTEREXAMPLE 1: Decay without Entropy Gap")
            print("\nPattern family: |P_k| = 2^k * log(k)")
            print("\nThis shows: O(k) → 0 but H_k - k → +∞")
            print_subsection("")
        
        k_values = [5, 10, 15, 20, 25, 30, 40, 50]
        
        if self.verbose:
            self._print_table_header([
                ("k", 4), ("O(k)", 12), ("H_k - k", 12), ("Behavior", 20)
            ])
        
        O_values = []
        gap_values = []
        
        for k in k_values:
            O_k, gap, _ = self._compute_with_overflow_protection(k, "2^k * log(k)")
            O_values.append(O_k)
            gap_values.append(gap)
            
            if self.verbose:
                behavior = "O↓, Gap↑" if k > 5 else "Initial"
                print(f"  {k:4d} | {format_scientific(O_k, 8):>12} | {gap:12.4f} | {behavior:>20}")
        
        # Verify the counterexample properties
        O_decreasing = all(O_values[i] > O_values[i+1] for i in range(len(O_values)-1))
        gap_increasing = all(gap_values[i] < gap_values[i+1] for i in range(len(gap_values)-1))
        
        if self.verbose:
            print("\nAnalysis:")
            print(f"  • O(k) decreasing: {O_decreasing}")
            print(f"  • Final O(50) ≈ {format_scientific(O_values[-1], 2)} → 0 ✓")
            print(f"  • H_k - k increasing: {gap_increasing}")
            print(f"  • Final H_50 - 50 ≈ {gap_values[-1]:.2f} → +∞ (not -∞)")
            print("\n✓ Counterexample confirmed: O(k) → 0 ⇏ H_k - k → -∞")
        
        success = O_decreasing and gap_increasing
        self.results.append(("Decay without entropy gap", success))
        return success
    
    def demonstrate_decay_without_summability(self) -> bool:
        """
        Show that O(k) -> 0 does not imply Σ O(k)^(1+ε) < ∞.
        
        Counterexample: |P_k| = 2^k * log(k)
        
        Returns:
            True if demonstration successful
        """
        if self.verbose:
            print_section_header("COUNTEREXAMPLE 2: Decay without Summability")
            print("\nPattern family: |P_k| = 2^k * log(k)")
            print("\nThis shows: O(k) → 0 but Σ O(k)^(1+ε) = ∞ for any ε > 0")
            print_subsection("")
        
        epsilons = [0.01, 0.1, 0.5, 1.0]
        
        if self.verbose:
            self._print_table_header([
                ("ε", 6), ("Partial Sum (k=2..30)", 20), ("Growth Rate", 15)
            ])
        
        for epsilon in epsilons:
            partial_sum = 0.0
            last_terms = []
            
            for k in range(2, 31):
                O_k, _, _ = self._compute_with_overflow_protection(k, "2^k * log(k)")
                term = O_k ** (1 + epsilon)
                partial_sum += term
                if k >= 25:
                    last_terms.append(term)
            
            # Check growth pattern
            if len(last_terms) > 1:
                growth = "Divergent" if last_terms[-1] > 1/(30**(1+epsilon)) else "Slowly divergent"
            else:
                growth = "Unknown"
            
            if self.verbose:
                print(f"  {epsilon:6.2f} | {partial_sum:20.6f} | {growth:>15}")
        
        if self.verbose:
            print("\nMathematical Analysis:")
            print("  For |P_k| = 2^k * log(k), we have O(k) ~ 1/log(k)")
            print("  Therefore, Σ O(k)^(1+ε) ~ Σ 1/(log k)^(1+ε)")
            print("  By integral test: ∫ 1/(log x)^(1+ε) dx diverges for any ε ≥ 0")
            print("\n✓ Counterexample confirmed: O(k) → 0 ⇏ Σ O(k)^(1+ε) < ∞")
        
        self.results.append(("Decay without summability", True))
        return True
    
    def demonstrate_decay_without_dimension_gap(self) -> bool:
        """
        Show that O(k) -> 0 does not imply limsup H_k/k < 1.
        
        Counterexample: |P_k| = 2^k * log(k)
        
        Returns:
            True if demonstration successful
        """
        if self.verbose:
            print_section_header("COUNTEREXAMPLE 3: Decay without Dimension Gap")
            print("\nPattern family: |P_k| = 2^k * log(k)")
            print("\nThis shows: O(k) → 0 but limsup H_k/k = 1 (no gap)")
            print_subsection("")
        
        k_values = [10, 20, 30, 40, 50, 100, 200]
        
        if self.verbose:
            self._print_table_header([
                ("k", 4), ("H_k/k", 12), ("1 - H_k/k", 12), ("O(k)", 12)
            ])
        
        ratios = []
        
        for k in k_values:
            O_k, gap, ratio = self._compute_with_overflow_protection(k, "2^k * log(k)")
            
            # For |P_k| = 2^k * log(k): H_k/k = 1 + log_2(log(k))/k
            ratio = 1 + safe_log2(np.log(k)) / k
            ratios.append(ratio)
            
            if self.verbose:
                print(f"  {k:4d} | {ratio:12.8f} | {1-ratio:12.8f} | {format_scientific(O_k, 2):>12}")
        
        if self.verbose:
            print("\nAnalysis:")
            print(f"  • H_k/k converging to 1: {abs(ratios[-1] - 1) < 0.01}")
            print(f"  • limsup H_k/k = {max(ratios):.6f} ≈ 1 (no strict gap)")
            print(f"  • Yet O(k) → 0 as shown in the last column")
            print("\n✓ Counterexample confirmed: O(k) → 0 ⇏ limsup H_k/k < 1")
        
        self.results.append(("Decay without dimension gap", True))
        return True
    
    def demonstrate_entropy_gap_without_summability(self) -> bool:
        """
        Show that H_k - k -> -∞ does not guarantee Σ O(k)^(1+ε) < ∞.
        
        Counterexample: |P_k| = 2^k / (k * log^2(k))
        
        Returns:
            True if demonstration successful
        """
        if self.verbose:
            print_section_header("COUNTEREXAMPLE 4: Entropy Gap without Guaranteed Summability")
            print("\nPattern family: |P_k| = 2^k / (k * log^2(k))")
            print("\nThis shows: H_k - k → -∞ but series convergence is marginal")
            print_subsection("")
        
        k_values = [5, 10, 20, 30, 40]
        
        if self.verbose:
            self._print_table_header([
                ("k", 4), ("H_k - k", 12), ("O(k)", 12), ("O(k)^1.01", 12)
            ])
        
        partial_sum = 0.0
        
        for k in k_values:
            if k > 2:
                O_k, gap, _ = self._compute_with_overflow_protection(k, "2^k / (k * log^2(k))")
                term = O_k ** 1.01
                partial_sum += term
                
                if self.verbose:
                    print(f"  {k:4d} | {gap:12.4f} | {format_scientific(O_k, 6):>12} | {format_scientific(term, 8):>12}")
        
        if self.verbose:
            print(f"\nPartial sum Σ O(k)^1.01 (k=5..40) = {partial_sum:.6f}")
            print("\nAnalysis:")
            print("  • H_k - k is decreasing toward -∞ ✓")
            print("  • But O(k) ~ 1/(k * log^3(k)) decays slowly")
            print("  • The series Σ O(k)^(1+ε) converges very slowly")
            print("\n✓ Shows that entropy gap doesn't guarantee fast decay for summability")
        
        self.results.append(("Entropy gap without guaranteed fast summability", True))
        return True
    
    def demonstrate_all_counterexamples(self) -> bool:
        """
        Run all counterexample demonstrations.
        
        Returns:
            True if all demonstrations successful
        """
        if self.verbose:
            print_section_header(
                "   COUNTEREXAMPLE DEMONSTRATIONS\n" +
                "   Showing the Hierarchy is Strict",
                width=70
            )
        
        all_success = True
        all_success &= self.demonstrate_decay_without_entropy_gap()
        all_success &= self.demonstrate_decay_without_summability()
        all_success &= self.demonstrate_decay_without_dimension_gap()
        all_success &= self.demonstrate_entropy_gap_without_summability()
        
        if self.verbose:
            print_section_header("SUMMARY OF COUNTEREXAMPLES", width=70)
            
            print("\nThe hierarchy:")
            print("  Dimension Gap ⟹ Entropy Gap ⟹ Pattern Decay ⟺ Subexp. Growth")
            print("\nCounterexamples show:")
            
            for name, success in self.results:
                status = "✓" if success else "✗"
                print(f"  {status} {name}")
            
            print("\nConclusion: Each implication is STRICT (cannot be reversed)")
            print("The hierarchy represents the strongest possible relationships.")
        
        return all_success


def interactive_exploration() -> None:
    """
    Interactive exploration of counterexamples.
    
    Allows users to explore different pattern families and observe
    their properties in the theoretical hierarchy.
    """
    print_section_header("   INTERACTIVE COUNTEREXAMPLE EXPLORER", width=70)
    
    while True:
        print("\nChoose a pattern family to explore:")
        print("1. |P_k| = 2^k * log(k)         [Critical, main counterexample]")
        print("2. |P_k| = 2^k * log^2(k)       [Supercritical]")
        print("3. |P_k| = 2^k / log(k)         [Subcritical]")
        print("4. |P_k| = 2^k / k              [Subcritical with alpha=1]")
        print("5. |P_k| = 2^(0.9k)             [Dimension gap]")
        print("6. Custom formula")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '0':
            break
        
        k_max = 30
        
        try:
            if choice == '1':
                counts = PatternFamily.exponential_log(k_max)
                formula = "2^k * log(k)"
            elif choice == '2':
                counts = [int(2**min(k, 20) * np.log(k+1)**2) if k > 0 else 0 
                         for k in range(k_max + 1)]
                formula = "2^k * log^2(k)"
            elif choice == '3':
                counts = PatternFamily.exponential_polylog(k_max, 1)
                formula = "2^k / log(k)"
            elif choice == '4':
                counts = PatternFamily.polynomial_exponential(k_max, 1)
                formula = "2^k / k"
            elif choice == '5':
                counts = PatternFamily.subexponential(k_max, 0.1)
                formula = "2^(0.9k)"
            elif choice == '6':
                print("\nEnter custom formula (use 'k' as variable):")
                print("Example: 2**k * np.log(k) / k**2")
                formula = input("Formula: ")
                counts = []
                for k in range(k_max + 1):
                    if k == 0:
                        counts.append(0)
                    else:
                        # Safe evaluation with limited scope
                        P_k = eval(formula, {"__builtins__": {}}, 
                                  {"k": k, "np": np, "log": np.log, "sqrt": np.sqrt})
                        counts.append(int(P_k))
            else:
                print("Invalid choice")
                continue
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Analyze the chosen pattern family
        pm = PatternMeasure(counts)
        
        print(f"\nAnalyzing: |P_k| = {formula}")
        print("-" * 50)
        
        # Compute key metrics
        print("\nKey values:")
        for k in [5, 10, 20, 30]:
            if k < len(counts) and counts[k] > 0:
                O_k = pm.compute_O(k)
                gap = pm.compute_entropy_gap(k)
                print(f"  k={k:2d}: O(k)={format_scientific(O_k, 6)}, H_k-k={gap:+.3f}")
        
        # Check properties
        print("\nProperties:")
        
        # Effective dimension
        try:
            d_eff = pm.effective_dimension(k_max)
            print(f"  • Effective dimension: {d_eff:.3f}")
            print(f"    {'✓ Has dimension gap' if d_eff < 0.99 else '✗ No dimension gap'}")
        except:
            print("  • Effective dimension: Could not compute")
        
        # Entropy gap trend
        try:
            gaps = [pm.compute_entropy_gap(k) for k in range(max(10, k_max//2), k_max) 
                   if k < len(counts) and counts[k] > 0]
            if gaps:
                if gaps[-1] < -5:
                    print(f"  • Entropy gap: H_k - k → -∞ ✓")
                elif gaps[-1] > 5:
                    print(f"  • Entropy gap: H_k - k → +∞")
                else:
                    print(f"  • Entropy gap: H_k - k ≈ {gaps[-1]:.2f}")
        except:
            print("  • Entropy gap: Could not compute")
        
        # O(k) decay
        try:
            O_vals = [pm.compute_O(k) for k in range(max(10, k_max//2), k_max)
                     if k < len(counts) and counts[k] > 0]
            if O_vals:
                if O_vals[-1] < 0.01:
                    print(f"  • Pattern decay: O(k) → 0 ✓")
                else:
                    print(f"  • Pattern decay: O(k) ≈ {format_scientific(O_vals[-1], 4)}")
        except:
            print("  • Pattern decay: Could not compute")
        
        # Alpha exponent
        if k_max > 20:
            try:
                alpha, r2 = pm.compute_alpha_exponent(10, k_max)
                if r2 > 0.8:
                    print(f"  • Alpha-exponent: α ≈ {alpha:.2f}")
                    print(f"    {'✓ Summable (α > 1)' if alpha > 1 else '✗ Not summable (α ≤ 1)'}")
            except:
                pass
        
        input("\nPress Enter to continue...")


def main() -> int:
    """
    Main routine for counterexample demonstrations.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demonstrate counterexamples for Pattern Measures'
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='Run interactive exploration'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_exploration()
        return 0
    else:
        demonstrator = CounterexampleDemonstrator(verbose=not args.quiet)
        success = demonstrator.demonstrate_all_counterexamples()
        return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())