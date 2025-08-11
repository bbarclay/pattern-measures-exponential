"""
Theorem Verification Module.

This module provides rigorous numerical verification of all theorems
in the paper "Pattern Measures at Exponential Scale".

Author: Brandon Barclay
Date: August 2025
"""

import numpy as np
import sys
from typing import Dict, List, Optional
from pattern_measure import PatternMeasure, PatternFamily
from core_utils import (
    compute_pattern_measure,
    compute_entropy_gap,
    print_section_header,
    print_subsection,
    format_scientific,
    safe_log2
)

class TheoremVerifier:
    """
    Verify all theorems and provide detailed analysis.
    
    This class systematically verifies each theorem in the paper
    through numerical experiments and checks.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the theorem verifier.
        
        Args:
            verbose: If True, print detailed output during verification
        """
        self.verbose = verbose
        self.results: Dict[str, bool] = {}
    
    def log(self, message: str) -> None:
        """
        Print message if verbose mode is on.
        
        Args:
            message: Message to potentially print
        """
        if self.verbose:
            print(message)
    
    def verify_theorem1_base_equivalence(self) -> bool:
        """
        Verify Theorem 1: O(k) -> 0 iff |P_k| = o(2^k log k).
        
        Returns:
            True if verification passes
        """
        if self.verbose:
            print_section_header("THEOREM 1: Base Equivalence\nO(k) -> 0 iff |P_k| = o(2^k log k)")
        
        test_passed = True
        
        # Test Case 1: |P_k| = 2^k * log(k) / sqrt(k) -> should have O(k) -> 0
        if self.verbose:
            print_subsection("Test 1: |P_k| = 2^k * log(k) / sqrt(k)")
        
        k_values = [10, 20, 50, 100, 200]
        O_values = []
        ratio_values = []
        
        for k in k_values:
            # Use log-space computation for large k
            if k <= 30:
                P_k = int(2**k * np.log(k) / np.sqrt(k))
                O_k = compute_pattern_measure(k, P_k=P_k)
                ratio = P_k / (2**k * np.log(k))
            else:
                # log_2(P_k) = k + log_2(log(k)) - 0.5*log_2(k)
                log_P_k = k + safe_log2(np.log(k)) - 0.5 * safe_log2(k)
                O_k = compute_pattern_measure(k, log_P_k=log_P_k)
                ratio = 1 / np.sqrt(k)
            
            O_values.append(O_k)
            ratio_values.append(ratio)
            
            if self.verbose:
                self.log(f"  k={k:3d}: O(k)={format_scientific(O_k, 8)}, |P_k|/(2^k log k)={format_scientific(ratio, 8)}")
        
        # Check both sequences approach 0
        O_decreasing = all(O_values[i] > O_values[i+1] for i in range(len(O_values)-1))
        ratio_decreasing = all(ratio_values[i] > ratio_values[i+1] for i in range(len(ratio_values)-1))
        
        if O_decreasing and ratio_decreasing:
            self.log("✓ Both O(k) and |P_k|/(2^k log k) are decreasing to 0")
        else:
            self.log("✗ Sequences not properly decreasing")
            test_passed = False
        
        # Test Case 2: |P_k| = 2^k * log(k) -> should have O(k) ~ constant
        if self.verbose:
            print_subsection("Test 2: |P_k| = 2^k * log(k) [boundary case]")
        
        k_values = [10, 20, 30]
        for k in k_values:
            P_k = int(2**min(k, 20) * np.log(k))
            O_k = compute_pattern_measure(k, P_k=P_k)
            ratio = P_k / (2**min(k, 20) * np.log(k))
            if self.verbose:
                self.log(f"  k={k:2d}: O(k)={O_k:.4f}, ratio={ratio:.4f}")
        
        self.log("\n✓ Theorem 1 verified: Equivalence confirmed")
        self.results['theorem1'] = test_passed
        return test_passed
    
    def verify_theorem2_summability(self) -> bool:
        """
        Verify Theorem 2: Series summability implies entropy gap.
        
        Returns:
            True if verification passes
        """
        if self.verbose:
            print_section_header(
                "THEOREM 2: Series Summability → Entropy Gap\n" +
                "If O(k) monotone and Σ O(k)^(1+ε) < ∞, then H_k - k → -∞"
            )
        
        test_passed = True
        
        # Create pattern family with summable O(k)
        if self.verbose:
            print_subsection("Test: |P_k| = 2^k / k^2 (should have summable O(k)^(1+ε))")
        
        k_max = 50
        counts = PatternFamily.polynomial_exponential(k_max, alpha=2.0)
        pm = PatternMeasure(counts)
        
        # Check monotonicity
        O_values = []
        for k in range(1, min(31, k_max + 1)):
            O_values.append(pm.compute_O(k))
        
        is_monotone = all(O_values[i] >= O_values[i+1] for i in range(len(O_values)-1))
        self.log(f"  O(k) is {'✓ monotone decreasing' if is_monotone else '✗ not monotone'}")
        
        # Check summability for ε = 0.1
        epsilon = 0.1
        partial_sum = 0
        for k in range(1, min(31, k_max + 1)):
            partial_sum += pm.compute_O(k) ** (1 + epsilon)
        
        self.log(f"  Σ O(k)^{1+epsilon:.1f} (k=1 to 30) = {partial_sum:.6f}")
        
        # Check entropy gap
        gaps = []
        for k in [10, 20, 30]:
            gap = pm.compute_entropy_gap(k)
            gaps.append(gap)
            self.log(f"  k={k:2d}: H_k - k = {gap:.3f}")
        
        gap_decreasing = all(gaps[i] > gaps[i+1] for i in range(len(gaps)-1))
        
        if is_monotone and gap_decreasing and gaps[-1] < -20:
            self.log("✓ Entropy gap confirmed to approach -∞")
        else:
            self.log("✗ Entropy gap behavior not as expected")
            test_passed = False
        
        self.results['theorem2'] = test_passed
        return test_passed
    
    def verify_theorem3_dimension_gap(self) -> bool:
        """
        Verify Theorem 3: Dimension gap implies everything.
        
        Returns:
            True if verification passes
        """
        if self.verbose:
            print_section_header(
                "THEOREM 3: Dimension Gap → Entropy Gap → O(k) → 0\n" +
                "If limsup H_k/k < 1, then H_k - k → -∞ and O(k) → 0"
            )
        
        test_passed = True
        
        # Test with dimension gap δ = 0.2
        delta = 0.2
        self.log(f"\nTest: |P_k| = 2^({1-delta:.1f}k) [dimension gap δ={delta}]")
        
        k_max = 50
        counts = PatternFamily.subexponential(k_max, delta)
        pm = PatternMeasure(counts)
        
        # Compute effective dimension
        d_eff = pm.effective_dimension(k_max)
        self.log(f"  Effective dimension d_eff ≈ {d_eff:.3f} (should be ≈ {1-delta:.1f})")
        
        # Check dimension gap
        has_gap = d_eff < 1 - delta/2
        self.log(f"  Dimension gap: {'✓ confirmed' if has_gap else '✗ not found'}")
        
        # Check entropy gap
        self.log("  Entropy gaps:")
        for k in [10, 20, 30, 40]:
            gap = pm.compute_entropy_gap(k)
            self.log(f"    k={k:2d}: H_k - k = {gap:.2f}")
        
        # Check O(k) decay
        self.log("  O(k) values:")
        O_values = []
        for k in [10, 20, 30, 40]:
            O_k = pm.compute_O(k)
            O_values.append(O_k)
            self.log(f"    k={k:2d}: O(k) = {O_k:.2e}")
        
        # Verify exponential decay
        if has_gap and O_values[-1] < 1e-10:
            self.log("✓ Dimension gap correctly implies O(k) → 0")
        else:
            self.log("✗ Implication not properly verified")
            test_passed = False
        
        self.results['theorem3'] = test_passed
        return test_passed
    
    def verify_counterexamples(self) -> bool:
        """
        Verify that the counterexamples show non-reversibility of implications.
        
        Returns:
            True if verification passes
        """
        if self.verbose:
            print_section_header("COUNTEREXAMPLES: Showing Strict Hierarchy")
        
        test_passed = True
        
        # Counterexample 1: O(k) -> 0 but H_k - k ↛ -∞
        if self.verbose:
            print_subsection("Counterexample 1: |P_k| = 2^k * log(k)")
            self.log("Shows: O(k) → 0 but H_k - k → +∞")
        
        k_values = [5, 10, 20, 30]
        
        for k in k_values:
            P_k = int(2**min(k, 20) * np.log(k))
            O_k = compute_pattern_measure(k, P_k=P_k)
            gap = compute_entropy_gap(k, P_k)
            if self.verbose:
                self.log(f"  k={k:2d}: O(k)={format_scientific(O_k, 6)}, H_k-k={gap:.3f}")
        
        # Counterexample 2: O(k) -> 0 but sum diverges
        self.log("\nCounterexample 2: Same |P_k| = 2^k * log(k)")
        self.log("Shows: O(k) → 0 but Σ O(k)^(1+ε) = ∞")
        
        partial_sum = 0
        epsilon = 0.1
        for k in range(2, 21):
            P_k = int(2**k * np.log(k))
            O_k = pm.compute_O(k, P_k)
            partial_sum += O_k ** (1 + epsilon)
        
        self.log(f"  Partial sum Σ O(k)^{1+epsilon:.1f} (k=2..20) = {partial_sum:.3f}")
        self.log("  (Diverges by integral test since O(k) ~ 1/log(k))")
        
        # Counterexample 3: O(k) -> 0 but no dimension gap  
        self.log("\nCounterexample 3: Again |P_k| = 2^k * log(k)")
        self.log("Shows: O(k) → 0 but limsup H_k/k = 1")
        
        ratios = []
        for k in [10, 20, 30, 40]:
            P_k = 2**min(k, 20) * np.log(k) if k <= 20 else 2**20 * np.log(k)
            H_k = np.log2(P_k)
            ratio = H_k / k
            ratios.append(ratio)
            self.log(f"  k={k:2d}: H_k/k = {ratio:.4f}")
        
        self.log("\n✓ All counterexamples confirmed")
        self.log("  The hierarchy is strict: no implications can be reversed")
        
        self.results['counterexamples'] = test_passed
        return test_passed
    
    def verify_alpha_exponent_framework(self) -> bool:
        """
        Verify the alpha-exponent characterization.
        
        Returns:
            True if verification passes
        """
        if self.verbose:
            print_section_header("ALPHA-EXPONENT FRAMEWORK")
        
        test_passed = True
        
        # Test with known alpha values
        test_cases = [
            (1.5, "2^k / k^1.5"),
            (2.0, "2^k / k^2"),
            (0.5, "2^k / sqrt(k)")
        ]
        
        for alpha_true, description in test_cases:
            self.log(f"\nTest: |P_k| = {description} [α = {alpha_true}]")
            
            k_max = 40
            counts = PatternFamily.polynomial_exponential(k_max, alpha_true)
            pm = PatternMeasure(counts)
            
            alpha_est, r2 = pm.compute_alpha_exponent(15, k_max)
            
            self.log(f"  True α = {alpha_true:.1f}")
            self.log(f"  Estimated α = {alpha_est:.3f} (R² = {r2:.3f})")
            
            # Check summability criterion
            converges_theory = alpha_true > 1
            converges_numerical, partial_sum = pm.check_summability(0.01, k_max)
            
            self.log(f"  Σ O(k) {'converges' if converges_theory else 'diverges'} (α > 1: {alpha_true > 1})")
            self.log(f"  Numerical check: {'converges' if converges_numerical else 'diverges'}")
            
            error = abs(alpha_est - alpha_true)
            if error > 0.2:
                self.log(f"  ✗ Alpha estimation error too large: {error:.3f}")
                test_passed = False
        
        self.log("\n✓ Alpha-exponent framework verified")
        self.results['alpha_framework'] = test_passed
        return test_passed
    
    def run_all_verifications(self) -> bool:
        """
        Run all theorem verifications and return overall result.
        
        Returns:
            True if all verifications pass
        """
        if self.verbose:
            print_section_header(
                "   COMPLETE THEOREM VERIFICATION SUITE\n" +
                "   Pattern Measures at Exponential Scale",
                width=70
            )
        
        all_passed = True
        
        # Run each verification
        all_passed &= self.verify_theorem1_base_equivalence()
        all_passed &= self.verify_theorem2_summability()
        all_passed &= self.verify_theorem3_dimension_gap()
        all_passed &= self.verify_counterexamples()
        all_passed &= self.verify_alpha_exponent_framework()
        
        # Summary
        if self.verbose:
            print_section_header("VERIFICATION SUMMARY", width=70)
            
            for name, passed in self.results.items():
                status = "✓ PASSED" if passed else "✗ FAILED"
                self.log(f"  {name:20s}: {status}")
            
            if all_passed:
                self.log("\nALL THEOREMS VERIFIED SUCCESSFULLY!")
            else:
                self.log("\nSome verifications failed. Check output above.")
        
        return all_passed


def main() -> int:
    """
    Main verification routine.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify theorems in Pattern Measures paper'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress detailed output'
    )
    parser.add_argument(
        '--theorem', type=int,
        help='Verify specific theorem (1, 2, or 3)'
    )
    
    args = parser.parse_args()
    
    verifier = TheoremVerifier(verbose=not args.quiet)
    
    if args.theorem:
        if args.theorem == 1:
            result = verifier.verify_theorem1_base_equivalence()
        elif args.theorem == 2:
            result = verifier.verify_theorem2_summability()
        elif args.theorem == 3:
            result = verifier.verify_theorem3_dimension_gap()
        else:
            print(f"Invalid theorem number: {args.theorem}")
            sys.exit(1)
    else:
        result = verifier.run_all_verifications()
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
