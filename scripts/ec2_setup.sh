#!/usr/bin/env bash
# CARR-v2 EC2 setup script
# Tested on: Deep Learning AMI (Ubuntu 22.04) with CUDA 12.1
# Run once after launching the instance:
#   chmod +x scripts/ec2_setup.sh && ./scripts/ec2_setup.sh

set -euo pipefail

# ---- 1. System packages ----
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git screen htop

# ---- 2. Python virtual environment ----
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. PyTorch (CUDA 12.1 wheel) ----
pip install --upgrade pip wheel
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ---- 4. Project dependencies ----
pip install -r requirements.txt

# ---- 5. Install package in editable mode ----
pip install -e .

# ---- 6. Verify CUDA is visible ----
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train (200 epochs, 100k users, 20k items):"
echo "  screen -S carr python examples/mini_trainer.py"
echo ""
echo "To resume from checkpoint:"
echo "  python examples/mini_trainer.py --resume outputs/checkpoints/ckpt_epoch0020.pt"
echo ""
echo "To run ablation (4 policies × 100 epochs):"
echo "  screen -S ablation python examples/ablation_runner.py"
echo ""
echo "To use real data (MovieLens or Amazon TSV):"
echo "  python examples/mini_trainer.py --data_path /data/interactions.tsv"
