"""
Pattern Measure Core Implementation
Author: Brandon Barclay
Date: August 2025

This module implements the pattern measure O(k) and related functions
for analyzing pattern density at exponential scale.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
import warnings

class PatternMeasure:
    """
    A class for computing and analyzing pattern measures at exponential scale.
    
    The pattern measure O(k) = |P_k| / (2^k * log_2(k+1)) captures
    the density of pattern families relative to exponential growth.
    """
    
    def __init__(self, pattern_counts: Optional[List[int]] = None):
        """
        Initialize with optional pattern counts.
        
        Args:
            pattern_counts: List where pattern_counts[k] = |P_k|
        """
        self.pattern_counts = pattern_counts if pattern_counts else []
        
    def compute_O(self, k: int, P_k: Optional[int] = None) -> float:
        """
        Compute the pattern measure O(k).
        
        Args:
            k: Scale parameter
            P_k: Pattern count at scale k (uses stored value if not provided)
            
        Returns:
            O(k) = |P_k| / (2^k * log_2(k+1))
        """
        if P_k is None:
            if k >= len(self.pattern_counts):
                raise ValueError(f"No pattern count available for k={k}")
            P_k = self.pattern_counts[k]
            
        if k <= 0:
            raise ValueError("k must be positive")
            
        return P_k / (2**k * np.log2(k + 1))
    
    def compute_entropy(self, k: int, P_k: Optional[int] = None) -> float:
        """
        Compute the entropy H_k = log_2(|P_k|).
        
        Args:
            k: Scale parameter
            P_k: Pattern count at scale k
            
        Returns:
            H_k = log_2(|P_k|)
        """
        if P_k is None:
            if k >= len(self.pattern_counts):
                raise ValueError(f"No pattern count available for k={k}")
            P_k = self.pattern_counts[k]
            
        if P_k <= 0:
            return float('-inf')
            
        return np.log2(P_k)
    
    def compute_entropy_gap(self, k: int, P_k: Optional[int] = None) -> float:
        """
        Compute the entropy gap H_k - k.
        
        Args:
            k: Scale parameter
            P_k: Pattern count at scale k
            
        Returns:
            H_k - k = log_2(|P_k|/2^k)
        """
        H_k = self.compute_entropy(k, P_k)
        return H_k - k
    
    def effective_dimension(self, k_max: int) -> float:
        """
        Compute the effective dimension d_eff = limsup H_k/k.
        
        Args:
            k_max: Maximum k to consider
            
        Returns:
            Approximation of d_eff
        """
        if not self.pattern_counts or k_max > len(self.pattern_counts):
            raise ValueError("Insufficient pattern counts")
            
        ratios = []
        for k in range(1, min(k_max, len(self.pattern_counts))):
            if self.pattern_counts[k] > 0:
                H_k = np.log2(self.pattern_counts[k])
                ratios.append(H_k / k)
                
        if not ratios:
            return 0.0
            
        # Return maximum of later values (approximating limsup)
        n = len(ratios)
        return max(ratios[n//2:]) if n > 1 else ratios[0]
    
    def compute_alpha_exponent(self, k_start: int = 10, k_end: Optional[int] = None) -> Tuple[float, float]:
        """
        Estimate the alpha-exponent from the pattern counts.
        
        Alpha = lim_{k->inf} log_2(2^k/|P_k|) / log_2(k)
        
        Args:
            k_start: Starting k for estimation
            k_end: Ending k for estimation
            
        Returns:
            (alpha_estimate, r_squared) where r_squared measures fit quality
        """
        if not self.pattern_counts:
            raise ValueError("No pattern counts available")
            
        if k_end is None:
            k_end = len(self.pattern_counts) - 1
            
        k_values = []
        y_values = []
        
        for k in range(k_start, min(k_end + 1, len(self.pattern_counts))):
            if k > 0 and self.pattern_counts[k] > 0:
                # log_2(2^k/|P_k|) = k - log_2(|P_k|)
                numerator = k - np.log2(self.pattern_counts[k])
                denominator = np.log2(k)
                if denominator > 0:
                    k_values.append(np.log(k))
                    y_values.append(numerator / denominator)
        
        if len(k_values) < 2:
            return (0.0, 0.0)
        
        # Use later values for better asymptotic estimate
        n = len(k_values)
        k_values = k_values[n//2:]
        y_values = y_values[n//2:]
        
        # Estimate alpha as mean of later ratios
        alpha_est = np.mean(y_values)
        
        # Compute R^2 to measure consistency
        y_mean = np.mean(y_values)
        ss_tot = np.sum((np.array(y_values) - y_mean)**2)
        ss_res = np.sum((np.array(y_values) - alpha_est)**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return (alpha_est, r_squared)
    
    def check_summability(self, epsilon: float = 0.1, k_max: Optional[int] = None) -> Tuple[bool, float]:
        """
        Check if sum O(k)^(1+epsilon) converges.
        
        Args:
            epsilon: Exponent parameter
            k_max: Maximum k to sum up to
            
        Returns:
            (appears_convergent, partial_sum)
        """
        if not self.pattern_counts:
            raise ValueError("No pattern counts available")
            
        if k_max is None:
            k_max = len(self.pattern_counts) - 1
            
        partial_sum = 0
        last_term = 0
        
        for k in range(1, min(k_max + 1, len(self.pattern_counts))):
            O_k = self.compute_O(k)
            term = O_k ** (1 + epsilon)
            partial_sum += term
            last_term = term
            
        # Heuristic: convergent if terms are decreasing rapidly
        appears_convergent = last_term < 1e-6 and k_max > 20
        
        return (appears_convergent, partial_sum)


class PatternFamily:
    """
    Generate various pattern families for testing and demonstration.
    """
    
    @staticmethod
    def exponential_log(k_max: int, c: float = 1.0) -> List[int]:
        """
        Generate pattern counts |P_k| = c * 2^k * log(k).
        This is the critical case where O(k) ~ 1/log(k).
        """
        counts = [0]  # P_0 undefined
        for k in range(1, k_max + 1):
            counts.append(int(c * 2**k * np.log(k + 1)))
        return counts
    
    @staticmethod
    def exponential_polylog(k_max: int, alpha: float = 2.0) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k / (log(k))^alpha.
        Subcritical if alpha > 0.
        """
        counts = [0]
        for k in range(1, k_max + 1):
            counts.append(int(2**k / (np.log(k + 1)**alpha)))
        return counts
    
    @staticmethod
    def subexponential(k_max: int, delta: float = 0.1) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^((1-delta)k).
        Has dimension gap with d_eff = 1 - delta < 1.
        """
        counts = [0]
        for k in range(1, k_max + 1):
            counts.append(int(2**((1 - delta) * k)))
        return counts
    
    @staticmethod
    def polynomial_exponential(k_max: int, alpha: float = 1.5) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k / k^alpha.
        """
        counts = [0]
        for k in range(1, k_max + 1):
            counts.append(int(2**k / (k**alpha)))
        return counts
    
    @staticmethod
    def full_exponential(k_max: int) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k.
        Maximal pattern family.
        """
        counts = [0]
        for k in range(1, k_max + 1):
            counts.append(2**k)
        return counts


def verify_base_equivalence(k_values: List[int], threshold: float = 0.01) -> bool:
    """
    Verify Theorem 1: O(k) -> 0 iff |P_k| = o(2^k log k).
    
    Args:
        k_values: Values of k to test
        threshold: Threshold for "approaching zero"
        
    Returns:
        True if the equivalence holds empirically
    """
    # Test with |P_k| = 2^k * log(k) / k (should have O(k) -> 0)
    pm = PatternMeasure()
    
    O_values = []
    ratio_values = []
    
    for k in k_values:
        if k > 1:
            P_k = int(2**k * np.log(k) / k)
            O_k = pm.compute_O(k, P_k)
            O_values.append(O_k)
            
            # Check if |P_k| / (2^k log k) -> 0
            ratio = P_k / (2**k * np.log(k))
            ratio_values.append(ratio)
    
    # Both should approach 0
    O_approaches_zero = O_values[-1] < threshold
    ratio_approaches_zero = ratio_values[-1] < threshold
    
    return O_approaches_zero == ratio_approaches_zero


def verify_dimension_gap_implication(delta: float = 0.1, k_max: int = 50) -> bool:
    """
    Verify Theorem 3: Dimension gap implies entropy gap and O(k) -> 0.
    
    Args:
        delta: Dimension gap parameter
        k_max: Maximum k to test
        
    Returns:
        True if the implication holds
    """
    # Create pattern family with dimension gap
    counts = PatternFamily.subexponential(k_max, delta)
    pm = PatternMeasure(counts)
    
    # Check effective dimension
    d_eff = pm.effective_dimension(k_max)
    has_dimension_gap = d_eff < 1 - delta/2  # Should be approximately 1-delta
    
    # Check entropy gap
    entropy_gaps = []
    for k in range(k_max//2, k_max + 1):
        gap = pm.compute_entropy_gap(k)
        entropy_gaps.append(gap)
    
    has_entropy_gap = entropy_gaps[-1] < -10  # Should go to -infinity
    
    # Check O(k) -> 0
    O_values = []
    for k in range(k_max//2, k_max + 1):
        O_values.append(pm.compute_O(k))
    
    O_vanishes = O_values[-1] < 1e-6
    
    # Verify implication chain
    if has_dimension_gap:
        return has_entropy_gap and O_vanishes
    return True  # Vacuously true if no dimension gap


if __name__ == "__main__":
    # Demo: Create and analyze different pattern families
    print("Pattern Measure Analysis Demo")
    print("=" * 50)
    
    k_max = 30
    
    # Critical case: |P_k| = 2^k * log(k)
    print("\n1. Critical Pattern Family: |P_k| = 2^k * log(k)")
    counts = PatternFamily.exponential_log(k_max)
    pm = PatternMeasure(counts)
    
    for k in [5, 10, 20, 30]:
        O_k = pm.compute_O(k)
        gap = pm.compute_entropy_gap(k)
        print(f"  k={k:2d}: O(k)={O_k:.6f}, H_k-k={gap:.3f}")
    
    alpha, r2 = pm.compute_alpha_exponent(10, k_max)
    print(f"  Alpha-exponent: {alpha:.3f} (R²={r2:.3f})")
    
    # Subcritical case
    print("\n2. Subcritical: |P_k| = 2^k / (log k)^2")
    counts = PatternFamily.exponential_polylog(k_max, alpha=2.0)
    pm = PatternMeasure(counts)
    
    for k in [5, 10, 20, 30]:
        O_k = pm.compute_O(k)
        print(f"  k={k:2d}: O(k)={O_k:.6f}")
    
    converges, sum_val = pm.check_summability(epsilon=0.1, k_max=k_max)
    print(f"  Sum O(k)^1.1 {'converges' if converges else 'diverges'} (partial sum: {sum_val:.3f})")
    
    # Verify theorems
    print("\n3. Theorem Verification")
    print(f"  Base equivalence holds: {verify_base_equivalence(list(range(10, 51)))}")
    print(f"  Dimension gap implication holds: {verify_dimension_gap_implication()}")
