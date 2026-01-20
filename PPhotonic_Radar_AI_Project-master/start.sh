#!/bin/bash
# Start the Photonic Radar AI project

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Photonic Radar AI - Startup"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python environment...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
echo ""

# Check core validation
echo -e "${BLUE}Validating core...${NC}"
if python3 run_core.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Core validated"
else
    echo -e "${YELLOW}⚠${NC} Core has warnings (continuing anyway)"
fi
echo ""

# Show menu
echo -e "${BLUE}Available options:${NC}"
echo "  1) Start Web UI (Streamlit)"
echo "  2) Run Training (main.py)"
echo "  3) Console Mode (app_console.py)"
echo "  4) Show Status"
echo "  5) Exit"
echo ""

# Default to web UI if no argument
MODE=${1:-1}

case $MODE in
    1|web|ui|streamlit)
        echo -e "${GREEN}Starting Web UI...${NC}"
        echo ""
        python3 launcher.py
        ;;
    2|train|main)
        echo -e "${GREEN}Starting Training Mode...${NC}"
        echo ""
        python3 main.py
        ;;
    3|console|cli)
        echo -e "${GREEN}Starting Console Mode...${NC}"
        echo ""
        python3 app_console.py
        ;;
    4|status)
        echo -e "${GREEN}Application Status${NC}"
        echo ""
        python3 core_cli.py status
        ;;
    5|exit|quit)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid option. Starting Web UI...${NC}"
        echo ""
        python3 launcher.py
        ;;
esac
