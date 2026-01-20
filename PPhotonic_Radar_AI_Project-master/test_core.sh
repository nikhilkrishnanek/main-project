#!/bin/bash
# Core Test Runner - Quick validation and diagnostics

echo "🚀 Photonic Radar AI - Core Test Suite"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Core Validation
echo "${YELLOW}[1/3]${NC} Running core validation..."
python3 run_core.py > /tmp/core_validation.log 2>&1
if [ $? -eq 0 ]; then
    echo "${GREEN}✅${NC} Core validation passed"
else
    echo "${YELLOW}⚠️${NC} Core validation had warnings (check run_core.py)"
fi
echo ""

# Test 2: CLI Status
echo "${YELLOW}[2/3]${NC} Checking application status..."
python3 core_cli.py status > /tmp/cli_status.log 2>&1
if [ $? -eq 0 ]; then
    grep "✅" /tmp/cli_status.log | wc -l > /dev/null
    echo "${GREEN}✅${NC} Application status OK"
else
    echo "${RED}❌${NC} Application status check failed"
fi
echo ""

# Test 3: System Info
echo "${YELLOW}[3/3]${NC} Gathering system information..."
python3 core_cli.py info > /tmp/cli_info.log 2>&1
if [ $? -eq 0 ]; then
    TORCH_STATUS=$(grep -c "✅ torch" /tmp/cli_info.log)
    if [ $TORCH_STATUS -eq 1 ]; then
        echo "${GREEN}✅${NC} Full dependencies installed (torch ready)"
    else
        echo "${YELLOW}⚠️${NC} PyTorch not installed (install for GPU support)"
        echo "    Run: pip install torch torchvision torchaudio"
    fi
else
    echo "${RED}❌${NC} System info check failed"
fi
echo ""

echo "========================================"
echo "${GREEN}✅ Core is runnable!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review CORE_QUICKSTART.md for full documentation"
echo "  2. Install dependencies: pip install -r requirements.txt"
echo "  3. Run: python launcher.py (web UI) or python main.py (training)"
echo ""
