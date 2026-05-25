#!/usr/bin/env python3
"""
Hyperlinearity alpha calculation script
Fits the power law: k* = c * d^alpha  =>  log(k*) = log(c) + alpha * log(d)
across model scales (Qwen2.5 1.5B, 3B, 7B).
"""

import math
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate scaling exponent alpha for neuron bypass thresholds.")
    parser.add_argument(
        "--k3b", type=float, default=None,
        help="The bypass threshold (k*) observed for Qwen2.5 3B (d=2048). If not provided, fits on 1.5B and 7B only."
    )
    args = parser.parse_args()

    # Known dimensions
    d_15b = 1536
    d_3b = 2048
    d_7b = 4096
    d_70b = 8192

    # Known k* values
    k_15b = 200
    k_7b = 2500

    print("=" * 60)
    print("  Contrastive Neuron Attribution (CNA) Bypass Scaling Law  ")
    print("=" * 60)

    if args.k3b is None:
        print("[*] No 3B data point provided. Fitting on two points (1.5B and 7B)...")
        # Direct calculation
        # log(k_7b) - log(k_15b) = alpha * (log(d_7b) - log(d_15b))
        alpha = (math.log(k_7b) - math.log(k_15b)) / (math.log(d_7b) - math.log(d_15b))
        c = k_15b / (d_15b ** alpha)
        
        # Predict 3B threshold under this fit
        k_3b_pred = c * (d_3b ** alpha)
        # Predict 70B threshold under this fit
        k_70b_pred = c * (d_70b ** alpha)

        print("\nFIT RESULTS:")
        print(f"  Exponent (alpha): {alpha:.4f}")
        print(f"  Constant (c):     {c:.6e}")
        print("-" * 40)
        print("PREDICTIONS:")
        print(f"  Qwen2.5 3B (d=2048)  -> predicted k* ~ {k_3b_pred:.1f} (run qwen3b_testing.py to verify!)")
        print(f"  Qwen2.5 70B (d=8192) -> predicted k* ~ {k_70b_pred:.1f}")

    else:
        k_3b = args.k3b
        print(f"[+] 3B data point provided: k* = {k_3b}")
        print("[*] Performing log-log linear regression on 3 data points (1.5B, 3B, 7B)...")

        # X = log(d), Y = log(k*)
        X = [math.log(d_15b), math.log(d_3b), math.log(d_7b)]
        Y = [math.log(k_15b), math.log(k_3b), math.log(k_7b)]

        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(Y) / n

        # Fit line: y = m*x + b => Y = alpha * X + log(c)
        numerator = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
        denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            print("[-] Error: calculation denominator is zero.")
            return

        alpha = numerator / denominator
        log_c = mean_y - alpha * mean_x
        c = math.exp(log_c)

        # R^2 calculation
        y_pred = [alpha * x + log_c for x in X]
        ss_tot = sum((y - mean_y) ** 2 for y in Y)
        ss_res = sum((Y[i] - y_pred[i]) ** 2 for i in range(n))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        # Predict 70B
        k_70b_pred = c * (d_70b ** alpha)

        print("\nFIT RESULTS:")
        print(f"  Exponent (alpha): {alpha:.4f}")
        print(f"  Constant (c):     {c:.6e}")
        print(f"  R^2 Coefficient:  {r2:.6f}")
        print("-" * 40)
        print("EMPIRICAL VS PREDICTED VALUES:")
        print(f"  1.5B (d=1536) -> Empirical k* = {k_15b:<6} | Fitted k* ~ {c * (d_15b ** alpha):.1f}")
        print(f"  3.0B (d=2048) -> Empirical k* = {k_3b:<6} | Fitted k* ~ {c * (d_3b ** alpha):.1f}")
        print(f"  7.0B (d=4096) -> Empirical k* = {k_7b:<6} | Fitted k* ~ {c * (d_7b ** alpha):.1f}")
        print("-" * 40)
        print("EXTRAPOLATIONS:")
        print(f"  Qwen2.5 70B (d=8192) -> predicted k* ~ {k_70b_pred:.1f} neurons")

        # Interpret fit quality
        print("\nINTERPRETATION:")
        if r2 > 0.99:
            print("  [OK] Outstanding fit! The power law scaling model holds exceptionally well.")
        elif r2 > 0.95:
            print("  [OK] Strong fit! The power law model describes the scaling behavior well.")
        else:
            print("  [!] Moderate fit. The scaling behavior might deviate from a pure power law.")

    print("=" * 60)

if __name__ == "__main__":
    main()
