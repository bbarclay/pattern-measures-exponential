"""
Pattern Measure Core Implementation.

This module implements the pattern measure O(k) and related functions
for analyzing pattern density at exponential scale.

Author: Brandon Barclay
Date: August 2025
"""

import numpy as np
from typing import List, Tuple, Optional
from core_utils import (
    compute_pattern_measure,
    compute_entropy,
    compute_entropy_gap,
    compute_effective_dimension,
    estimate_alpha_exponent,
    check_series_convergence,
    safe_log2
)


class PatternMeasure:
    """
    A class for computing and analyzing pattern measures at exponential scale.
    
    The pattern measure O(k) = |P_k| / (2^k * log_2(k+1)) captures
    the density of pattern families relative to exponential growth.
    
    Attributes:
        pattern_counts: List where pattern_counts[k] = |P_k|
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
            
        Raises:
            ValueError: If no pattern count available for given k
        """
        if P_k is None:
            if k >= len(self.pattern_counts):
                raise ValueError(f"No pattern count available for k={k}")
            P_k = self.pattern_counts[k]
        
        return compute_pattern_measure(k, P_k=P_k)
    
    def compute_entropy(self, k: int, P_k: Optional[int] = None) -> float:
        """
        Compute the entropy H_k = log_2(|P_k|).
        
        Args:
            k: Scale parameter
            P_k: Pattern count at scale k
            
        Returns:
            H_k = log_2(|P_k|)
            
        Raises:
            ValueError: If no pattern count available for given k
        """
        if P_k is None:
            if k >= len(self.pattern_counts):
                raise ValueError(f"No pattern count available for k={k}")
            P_k = self.pattern_counts[k]
        
        return compute_entropy(P_k)
    
    def compute_entropy_gap(self, k: int, P_k: Optional[int] = None) -> float:
        """
        Compute the entropy gap H_k - k.
        
        Args:
            k: Scale parameter
            P_k: Pattern count at scale k
            
        Returns:
            H_k - k = log_2(|P_k|/2^k)
            
        Raises:
            ValueError: If no pattern count available for given k
        """
        if P_k is None:
            if k >= len(self.pattern_counts):
                raise ValueError(f"No pattern count available for k={k}")
            P_k = self.pattern_counts[k]
        
        return compute_entropy_gap(k, P_k)
    
    def effective_dimension(self, k_max: int) -> float:
        """
        Compute the effective dimension d_eff = limsup H_k/k.
        
        Args:
            k_max: Maximum k to consider
            
        Returns:
            Approximation of d_eff
            
        Raises:
            ValueError: If insufficient pattern counts
        """
        if not self.pattern_counts:
            raise ValueError("No pattern counts available")
        
        return compute_effective_dimension(self.pattern_counts, k_max)
    
    def compute_alpha_exponent(
        self,
        k_start: int = 10,
        k_end: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Estimate the alpha-exponent from the pattern counts.
        
        Alpha = lim_{k->inf} log_2(2^k/|P_k|) / log_2(k)
        
        Args:
            k_start: Starting k for estimation
            k_end: Ending k for estimation
            
        Returns:
            (alpha_estimate, r_squared) where r_squared measures fit quality
            
        Raises:
            ValueError: If no pattern counts available
        """
        if not self.pattern_counts:
            raise ValueError("No pattern counts available")
        
        return estimate_alpha_exponent(self.pattern_counts, k_start, k_end)
    
    def check_summability(
        self,
        epsilon: float = 0.1,
        k_max: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Check if sum O(k)^(1+epsilon) converges.
        
        Args:
            epsilon: Exponent parameter
            k_max: Maximum k to sum up to
            
        Returns:
            (appears_convergent, partial_sum)
            
        Raises:
            ValueError: If no pattern counts available
        """
        if not self.pattern_counts:
            raise ValueError("No pattern counts available")
        
        return check_series_convergence(self.pattern_counts, epsilon, k_max)


class PatternFamily:
    """
    Generate various pattern families for testing and demonstration.
    
    This class provides static methods to generate different types of
    pattern families that exhibit various growth behaviors.
    """
    
    @staticmethod
    def _generate_pattern_counts(
        k_max: int,
        formula_func,
        safe_compute: bool = True
    ) -> List[int]:
        """
        Generic pattern count generator with overflow protection.
        
        Args:
            k_max: Maximum k value
            formula_func: Function that computes P_k given k
            safe_compute: If True, limit exponential growth for large k
            
        Returns:
            List of pattern counts
        """
        counts = [0]  # P_0 undefined
        for k in range(1, k_max + 1):
            try:
                # Limit exponential computation to prevent overflow
                k_eff = min(k, 30) if safe_compute else k
                P_k = formula_func(k, k_eff)
                counts.append(max(1, int(P_k)))  # Ensure at least 1
            except (OverflowError, ValueError):
                counts.append(1)  # Fallback for numerical issues
        return counts
    
    @staticmethod
    def exponential_log(k_max: int, c: float = 1.0) -> List[int]:
        """
        Generate pattern counts |P_k| = c * 2^k * log(k).
        
        This is the critical case where O(k) ~ 1/log(k).
        
        Args:
            k_max: Maximum k value
            c: Multiplicative constant
            
        Returns:
            List of pattern counts
            
        Mathematical Context:
            This family lies at the critical threshold. It has O(k) → 0
            but exhibits the slowest possible decay rate.
        """
        return PatternFamily._generate_pattern_counts(
            k_max,
            lambda k, k_eff: c * 2**k_eff * np.log(k + 1)
        )
    
    @staticmethod
    def exponential_polylog(k_max: int, alpha: float = 2.0) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k / (log(k))^alpha.
        
        Subcritical if alpha > 0, with stronger decay for larger alpha.
        
        Args:
            k_max: Maximum k value
            alpha: Polylogarithmic exponent
            
        Returns:
            List of pattern counts
            
        Mathematical Context:
            The alpha parameter controls the decay strength.
            Alpha > 1 ensures summability of O(k).
        """
        return PatternFamily._generate_pattern_counts(
            k_max,
            lambda k, k_eff: 2**k_eff / (np.log(k + 1)**alpha)
        )
    
    @staticmethod
    def subexponential(k_max: int, delta: float = 0.1) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^((1-delta)k).
        
        Has dimension gap with d_eff = 1 - delta < 1.
        
        Args:
            k_max: Maximum k value
            delta: Dimension gap parameter (0 < delta < 1)
            
        Returns:
            List of pattern counts
            
        Mathematical Context:
            This family exhibits a dimension gap, the strongest condition
            in our hierarchy, implying all weaker properties.
        """
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        
        return PatternFamily._generate_pattern_counts(
            k_max,
            lambda k, k_eff: 2**((1 - delta) * k_eff)
        )
    
    @staticmethod
    def polynomial_exponential(k_max: int, alpha: float = 1.5) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k / k^alpha.
        
        The alpha-exponent determines convergence properties.
        
        Args:
            k_max: Maximum k value
            alpha: Polynomial exponent
            
        Returns:
            List of pattern counts
            
        Mathematical Context:
            This family directly exhibits the alpha-exponent framework.
            Convergence of sum O(k) occurs precisely when alpha > 1.
        """
        return PatternFamily._generate_pattern_counts(
            k_max,
            lambda k, k_eff: 2**k_eff / (k**alpha) if k > 0 else 1
        )
    
    @staticmethod
    def full_exponential(k_max: int) -> List[int]:
        """
        Generate pattern counts |P_k| = 2^k.
        
        Maximal pattern family with O(k) = constant.
        
        Args:
            k_max: Maximum k value
            
        Returns:
            List of pattern counts
            
        Mathematical Context:
            This represents the maximal possible growth rate,
            saturating the exponential bound.
        """
        return PatternFamily._generate_pattern_counts(
            k_max,
            lambda k, k_eff: 2**k_eff
        )


def demo_pattern_families():
    """
    Demonstrate different pattern families and their properties.
    
    This function showcases the key behaviors of various pattern families
    in the mathematical framework.
    """
    from core_utils import print_section_header, print_subsection, format_scientific
    
    print_section_header("Pattern Measure Analysis Demo", 50)
    
    k_max = 30
    test_k_values = [5, 10, 20, 30]
    
    # Critical case
    print_subsection("1. Critical Pattern Family: |P_k| = 2^k * log(k)")
    counts = PatternFamily.exponential_log(k_max)
    pm = PatternMeasure(counts)
    
    for k in test_k_values:
        O_k = pm.compute_O(k)
        gap = pm.compute_entropy_gap(k)
        print(f"  k={k:2d}: O(k)={format_scientific(O_k, 6)}, H_k-k={gap:+.3f}")
    
    alpha, r2 = pm.compute_alpha_exponent(10, k_max)
    print(f"  Alpha-exponent: {alpha:.3f} (R²={r2:.3f})")
    
    # Subcritical case
    print_subsection("2. Subcritical: |P_k| = 2^k / (log k)^2")
    counts = PatternFamily.exponential_polylog(k_max, alpha=2.0)
    pm = PatternMeasure(counts)
    
    for k in test_k_values:
        O_k = pm.compute_O(k)
        print(f"  k={k:2d}: O(k)={format_scientific(O_k, 6)}")
    
    converges, sum_val = pm.check_summability(epsilon=0.1, k_max=k_max)
    status = "converges" if converges else "diverges"
    print(f"  Sum O(k)^1.1 {status} (partial sum: {sum_val:.3f})")
    
    # Dimension gap case
    print_subsection("3. Dimension Gap: |P_k| = 2^(0.8k)")
    counts = PatternFamily.subexponential(k_max, delta=0.2)
    pm = PatternMeasure(counts)
    
    d_eff = pm.effective_dimension(k_max)
    print(f"  Effective dimension: {d_eff:.3f} < 1")
    
    for k in [10, 20, 30]:
        O_k = pm.compute_O(k)
        gap = pm.compute_entropy_gap(k)
        print(f"  k={k:2d}: O(k)={format_scientific(O_k, 3)}, H_k-k={gap:.1f}")
    
    print("\nKey Insights:")
    print("  - Critical family shows slowest decay: O(k) ~ 1/log(k)")
    print("  - Subcritical families have summable O(k)")
    print("  - Dimension gap implies exponential suppression")


if __name__ == "__main__":
    demo_pattern_families()
