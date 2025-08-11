"""
Core utilities for Pattern Measures at Exponential Scale.

This module provides shared mathematical functions and utilities used across
the pattern measure framework implementation.

Author: Brandon Barclay
Date: August 2025
"""

import math
import numpy as np
from typing import Union, Optional, Tuple, List
import warnings


def safe_log2(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute log base 2 with safe handling of edge cases.
    
    Args:
        x: Value or array to compute log2 of
        
    Returns:
        log2(x) or -inf for non-positive values
        
    Mathematical Context:
        Used throughout for entropy calculations H_k = log_2(|P_k|)
    """
    if isinstance(x, (int, float)):
        if x <= 0:
            return float('-inf')
        return np.log2(x) if hasattr(np, 'log2') else math.log(x) / math.log(2)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.log2(x)
            result[x <= 0] = float('-inf')
            return result


def compute_pattern_measure(
    k: int,
    P_k: Optional[int] = None,
    log_P_k: Optional[float] = None
) -> float:
    """
    Compute the pattern measure O(k) = |P_k| / (2^k * log_2(k+1)).
    
    This is the central quantity in our framework, measuring pattern density
    relative to exponential growth with logarithmic normalization.
    
    Args:
        k: Scale parameter (must be positive)
        P_k: Pattern count at scale k
        log_P_k: Log base 2 of pattern count (for numerical stability)
        
    Returns:
        O(k) value
        
    Raises:
        ValueError: If k <= 0 or if neither P_k nor log_P_k provided
        
    Mathematical Context:
        O(k) captures the density of pattern families relative to the
        exponential baseline 2^k, normalized by log_2(k+1) to identify
        the critical threshold.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    if log_P_k is not None:
        # Use log-space computation for numerical stability
        # O(k) = 2^(log_P_k - k) / log_2(k+1)
        return 2**(log_P_k - k) / safe_log2(k + 1)
    elif P_k is not None and P_k > 0:
        if k <= 30:  # Direct computation for small k
            return P_k / (2**k * safe_log2(k + 1))
        else:  # Use log-space for large k to avoid overflow
            return compute_pattern_measure(k, log_P_k=safe_log2(P_k))
    elif P_k == 0:
        return 0.0
    else:
        raise ValueError("Must provide either P_k or log_P_k")


def compute_entropy(P_k: Union[int, float]) -> float:
    """
    Compute the entropy H_k = log_2(|P_k|).
    
    Args:
        P_k: Pattern count
        
    Returns:
        Entropy value
        
    Mathematical Context:
        The entropy measures the logarithmic size of the pattern family,
        directly related to its information-theoretic complexity.
    """
    return safe_log2(P_k)


def compute_entropy_gap(k: int, P_k: Union[int, float]) -> float:
    """
    Compute the entropy gap H_k - k = log_2(|P_k|/2^k).
    
    Args:
        k: Scale parameter
        P_k: Pattern count
        
    Returns:
        Entropy gap value
        
    Mathematical Context:
        The entropy gap measures the deficit from maximal entropy k.
        Negative values indicate subexponential growth.
    """
    if P_k <= 0:
        return float('-inf')
    return safe_log2(P_k) - k


def compute_effective_dimension(
    pattern_counts: List[int],
    k_max: Optional[int] = None
) -> float:
    """
    Compute the effective dimension d_eff = limsup H_k/k.
    
    Args:
        pattern_counts: List where pattern_counts[k] = |P_k|
        k_max: Maximum k to consider (default: len(pattern_counts)-1)
        
    Returns:
        Approximation of effective dimension
        
    Mathematical Context:
        The effective dimension characterizes the asymptotic growth rate.
        Values < 1 indicate a dimension gap and imply strong decay properties.
    """
    if not pattern_counts:
        raise ValueError("Empty pattern counts")
    
    if k_max is None:
        k_max = len(pattern_counts) - 1
    else:
        k_max = min(k_max, len(pattern_counts) - 1)
    
    ratios = []
    for k in range(1, k_max + 1):
        if k < len(pattern_counts) and pattern_counts[k] > 0:
            H_k = safe_log2(pattern_counts[k])
            ratios.append(H_k / k)
    
    if not ratios:
        return 0.0
    
    # Return maximum of later values (approximating limsup)
    n = len(ratios)
    return max(ratios[n//2:]) if n > 1 else ratios[0]


def estimate_alpha_exponent(
    pattern_counts: List[int],
    k_start: int = 10,
    k_end: Optional[int] = None
) -> Tuple[float, float]:
    """
    Estimate the alpha-exponent from pattern counts.
    
    Alpha characterizes the decay rate: |P_k| ~ 2^k / k^alpha
    
    Args:
        pattern_counts: List where pattern_counts[k] = |P_k|
        k_start: Starting k for estimation
        k_end: Ending k for estimation
        
    Returns:
        (alpha_estimate, r_squared) where r_squared measures fit quality
        
    Mathematical Context:
        The alpha-exponent provides a quantitative measure of decay strength.
        Alpha > 1 implies summability of O(k), establishing a sharp threshold.
    """
    if not pattern_counts:
        raise ValueError("No pattern counts available")
    
    if k_end is None:
        k_end = len(pattern_counts) - 1
    else:
        k_end = min(k_end, len(pattern_counts) - 1)
    
    if k_start >= k_end:
        raise ValueError("k_start must be less than k_end")
    
    y_values = []
    
    for k in range(k_start, k_end + 1):
        if k < len(pattern_counts) and pattern_counts[k] > 0:
            # Alpha = log_2(2^k/|P_k|) / log_2(k)
            numerator = k - safe_log2(pattern_counts[k])
            denominator = safe_log2(k)
            if denominator > 0:
                y_values.append(numerator / denominator)
    
    if len(y_values) < 2:
        return (0.0, 0.0)
    
    # Use later values for better asymptotic estimate
    n = len(y_values)
    y_values = y_values[n//2:]
    
    # Estimate alpha as mean of later ratios
    alpha_est = np.mean(y_values)
    
    # Compute R^2 to measure consistency
    y_mean = np.mean(y_values)
    ss_tot = np.sum((np.array(y_values) - y_mean)**2)
    ss_res = np.sum((np.array(y_values) - alpha_est)**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return (alpha_est, r_squared)


def check_series_convergence(
    pattern_counts: List[int],
    epsilon: float = 0.1,
    k_max: Optional[int] = None
) -> Tuple[bool, float]:
    """
    Check if sum O(k)^(1+epsilon) appears to converge.
    
    Args:
        pattern_counts: List where pattern_counts[k] = |P_k|
        epsilon: Exponent parameter (>0)
        k_max: Maximum k to sum up to
        
    Returns:
        (appears_convergent, partial_sum)
        
    Mathematical Context:
        Series convergence is a key criterion in our hierarchy.
        Convergence implies strong decay properties and entropy gaps.
    """
    if not pattern_counts:
        raise ValueError("No pattern counts available")
    
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    if k_max is None:
        k_max = len(pattern_counts) - 1
    else:
        k_max = min(k_max, len(pattern_counts) - 1)
    
    partial_sum = 0.0
    last_term = 0.0
    terms = []
    
    for k in range(1, k_max + 1):
        if k < len(pattern_counts) and pattern_counts[k] > 0:
            O_k = compute_pattern_measure(k, P_k=pattern_counts[k])
            term = O_k ** (1 + epsilon)
            partial_sum += term
            terms.append(term)
            last_term = term
    
    if len(terms) < 10:
        # Not enough data to determine convergence
        return (False, partial_sum)
    
    # Check if terms are decreasing rapidly enough
    # Heuristic: convergent if last terms are very small and decreasing
    recent_terms = terms[-5:]
    decreasing = all(recent_terms[i] > recent_terms[i+1] 
                     for i in range(len(recent_terms)-1))
    small_enough = recent_terms[-1] < 1e-6
    
    appears_convergent = decreasing and small_enough and k_max > 20
    
    return (appears_convergent, partial_sum)


def format_scientific(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation if very small/large.
    
    Args:
        value: Number to format
        precision: Number of significant digits
        
    Returns:
        Formatted string
    """
    if abs(value) < 1e-4 or abs(value) > 1e4:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def print_section_header(title: str, width: int = 60):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of header line
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 40):
    """
    Print a formatted subsection header.
    
    Args:
        title: Subsection title
        width: Width of header line
    """
    print(f"\n{title}")
    print("-" * width)