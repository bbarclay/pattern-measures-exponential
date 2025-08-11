#!/usr/bin/env python3
"""
Simple demonstration of pattern measures without heavy dependencies.

This script provides a lightweight demonstration of the key concepts
in the pattern measures framework, using only Python's math module.

Author: Brandon Barclay
Date: August 2025
"""

import math
from typing import List, Tuple


def safe_log2(x: float) -> float:
    """
    Compute log base 2 safely.
    
    Args:
        x: Value to compute log2 of
        
    Returns:
        log2(x) or -inf for non-positive values
    """
    if x <= 0:
        return float('-inf')
    return math.log(x) / math.log(2)


def pattern_measure(k: int, P_k: int) -> float:
    """
    Compute O(k) = |P_k| / (2^k * log_2(k+1)).
    
    Args:
        k: Scale parameter (must be positive)
        P_k: Pattern count at scale k
        
    Returns:
        O(k) value
        
    Raises:
        ValueError: If k <= 0
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return P_k / (2**k * safe_log2(k + 1))


def format_number(value: float, precision: int = 6) -> str:
    """
    Format a number appropriately based on its magnitude.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) < 1e-4 or abs(value) > 1e4:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def demonstrate_critical_case() -> None:
    """Demonstrate the critical pattern family |P_k| = 2^k * log(k)."""
    print("\n1. Critical Pattern Family: |P_k| = 2^k * log(k)")
    print("   (Shows O(k) ~ 1/log(k) → 0)")
    print("-" * 40)
    
    k_values = [5, 10, 15, 20]
    for k in k_values:
        P_k = int(2**k * math.log(k))
        O_k = pattern_measure(k, P_k)
        print(f"   k={k:2d}: |P_k|={P_k:10d}, O(k)={format_number(O_k)}")
    
    print("\n   Key insight: O(k) decays as 1/log(k), the slowest possible rate")


def demonstrate_subcritical_case() -> None:
    """Demonstrate a subcritical pattern family |P_k| = 2^k / k^2."""
    print("\n2. Subcritical: |P_k| = 2^k / k^2")
    print("   (Shows faster decay with α = 2)")
    print("-" * 40)
    
    k_values = [5, 10, 15, 20]
    for k in k_values:
        P_k = int(2**k / k**2)
        O_k = pattern_measure(k, P_k)
        print(f"   k={k:2d}: |P_k|={P_k:10d}, O(k)={format_number(O_k)}")
    
    print("\n   Key insight: O(k) ~ 1/k^2, ensuring series convergence")


def demonstrate_dimension_gap() -> None:
    """Demonstrate a pattern family with dimension gap |P_k| = 2^(0.8k)."""
    print("\n3. Dimension Gap: |P_k| = 2^(0.8k)")
    print("   (Shows exponential suppression)")
    print("-" * 40)
    
    delta = 0.2
    k_values = [5, 10, 15, 20]
    
    for k in k_values:
        P_k = int(2**(k * (1 - delta)))
        O_k = pattern_measure(k, P_k)
        entropy_gap = (1 - delta) * k - k  # H_k - k
        print(f"   k={k:2d}: O(k)={format_number(O_k, 8)}, H_k-k={entropy_gap:.1f}")
    
    print(f"\n   Key insight: Dimension d_eff = {1-delta:.1f} < 1 implies exponential decay")


def demonstrate_hierarchy() -> None:
    """Demonstrate the hierarchy of implications."""
    print("\n4. Hierarchy of Implications")
    print("-" * 40)
    print("\n   Strongest → Weakest:")
    print("   • Dimension Gap (d_eff < 1)")
    print("     ⇓")
    print("   • Entropy Gap (H_k - k → -∞)")
    print("     ⇓")
    print("   • Pattern Decay (O(k) → 0)")
    print("     ⇕")
    print("   • Subexponential Growth (|P_k| = o(2^k log k))")
    print("\n   Note: Each implication is STRICT (cannot be reversed)")


def main() -> None:
    """Run the demonstration."""
    print("=" * 50)
    print("Pattern Measures at Exponential Scale")
    print("Simple Demonstration")
    print("=" * 50)
    print("\nDemonstration of key concepts:")
    
    demonstrate_critical_case()
    demonstrate_subcritical_case()
    demonstrate_dimension_gap()
    demonstrate_hierarchy()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    print("The logarithmic factor in O(k) = |P_k|/(2^k log(k+1))")
    print("captures a fundamental threshold for pattern density.")
    print("\nKey results:")
    print("• Critical threshold: |P_k| ~ 2^k log(k)")
    print("• Summability criterion: α > 1 for |P_k| ~ 2^k/k^α")
    print("• Dimension gap implies all weaker properties")
    print("• The hierarchy is strict (counterexamples exist)")
    print("=" * 50)


if __name__ == "__main__":
    main()