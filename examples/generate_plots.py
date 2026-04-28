"""
Generate all publication-quality figures from CARR-v2 experiment outputs.

Run after training completes:
  python examples/generate_plots.py
  python examples/generate_plots.py --out_dir outputs
"""

import argparse

from carr_v2.plotting import generate_all

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate CARR-v2 plots")
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()
    generate_all(out_dir=args.out_dir)
