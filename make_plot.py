#!/usr/bin/env python3
"""
make_plot.py — builds speedup_plot.png from results.csv

Expected results.csv columns (header row required):
    threads,T_seq,T_k

Usage:
    python3 make_plot.py results.csv speedup_plot.png

This script does NOT invent any numbers — it only reads what you put in
results.csv from your own real benchmark runs.
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} results.csv speedup_plot.png")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    ks, s_emp = [], []
    t_seq = None

    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row["threads"])
            tk = float(row["T_k"])
            if t_seq is None:
                t_seq = float(row["T_seq"])
            ks.append(k)
            s_emp.append(t_seq / tk)

    # Derive p from k=2 empirical speedup (worksheet Step in Phase 3)
    if 2 in ks:
        s2 = s_emp[ks.index(2)]
        p = 2 * (1 - 1 / s2)
    else:
        p = None

    s_theo = None
    if p is not None:
        s_theo = [1.0 / ((1 - p) + (p / k)) for k in ks]

    plt.figure(figsize=(7, 5))
    plt.plot(ks, s_emp, "o-", label="Empirical S_emp(k)")
    if s_theo is not None:
        plt.plot(ks, s_theo, "s--", label=f"Theoretical S_theo(k), p={p:.4f}")
    plt.plot(ks, ks, ":", color="gray", label="Linear ideal")
    plt.xlabel("Threads (k)")
    plt.ylabel("Speedup")
    plt.title("Amdahl Reality Gap: Empirical vs. Theoretical Speedup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if p is not None:
        print(f"Derived p (from k=2) = {p:.6f}")


if __name__ == "__main__":
    main()
