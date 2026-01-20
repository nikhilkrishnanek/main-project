#!/bin/bash
# Quick Start - Photonic Radar AI Core

set -e

echo "🚀 Photonic Radar AI - Core Quick Start"
echo "========================================"
echo ""

# Step 1: Validate
echo "Step 1: Validating core..."
python3 run_core.py > /dev/null 2>&1 && echo "✅ Core validated" || echo "⚠️  Review CORE_QUICKSTART.md for issues"
echo ""

# Step 2: Show status
echo "Step 2: Checking application status..."
python3 core_cli.py status 2>/dev/null | tail -4
echo ""

# Step 3: Next steps
echo "Step 3: Next steps"
echo "===================="
echo ""
echo "To use the full application, install dependencies:"
echo "  pip install -r requirements.txt"
echo ""
echo "Then run one of:"
echo "  python launcher.py      # Web UI at http://localhost:8501"
echo "  python main.py          # Training mode"
echo "  python app_console.py   # Console interface"
echo ""
echo "For more information, see:"
echo "  • CORE_QUICKSTART.md"
echo "  • CORE_RUNNABLE_SUMMARY.md"
echo ""
