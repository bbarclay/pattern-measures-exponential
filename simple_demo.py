#!/usr/bin/env python3
"""
Simple demonstration of pattern measures without numpy dependency
Author: Brandon Barclay
Date: August 2025
"""

import math

def pattern_measure(k, P_k):
    """Compute O(k) = |P_k| / (2^k * log_2(k+1))"""
    if k <= 0:
        raise ValueError("k must be positive")
    return P_k / (2**k * math.log2(k + 1))

def main():
    print("Pattern Measures at Exponential Scale")
    print("=" * 50)
    print("\nDemonstration of key concepts:")
    
    # Example 1: Critical case |P_k| = 2^k * log(k)
    print("\n1. Critical Pattern Family: |P_k| = 2^k * log(k)")
    print("   (Shows O(k) ~ 1/log(k) → 0)")
    print("-" * 40)
    
    for k in [5, 10, 15, 20]:
        P_k = int(2**k * math.log(k))
        O_k = pattern_measure(k, P_k)
        print(f"   k={k:2d}: |P_k|={P_k:10d}, O(k)={O_k:.6f}")
    
    # Example 2: Subcritical case
    print("\n2. Subcritical: |P_k| = 2^k / k^2")
    print("   (Shows faster decay)")
    print("-" * 40)
    
    for k in [5, 10, 15, 20]:
        P_k = int(2**k / k**2)
        O_k = pattern_measure(k, P_k)
        print(f"   k={k:2d}: |P_k|={P_k:10d}, O(k)={O_k:.6f}")
    
    # Example 3: Dimension gap
    print("\n3. Dimension Gap: |P_k| = 2^(0.8k)")
    print("   (Shows exponential suppression)")
    print("-" * 40)
    
    for k in [5, 10, 15, 20]:
        P_k = int(2**(0.8 * k))
        O_k = pattern_measure(k, P_k)
        entropy_gap = 0.8 * k - k  # H_k - k
        print(f"   k={k:2d}: O(k)={O_k:.8f}, H_k-k={entropy_gap:.1f}")
    
    print("\n" + "=" * 50)
    print("Key Insight: The logarithmic factor in O(k) captures")
    print("a fundamental threshold for pattern density at scale.")

if __name__ == "__main__":
    main()
