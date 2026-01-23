#!/bin/bash
#
# Quick start script for the Portfolio Simulator
#
# Usage:
#   ./run_portfolio_simulator.sh
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Portfolio Simulator - Quick Start${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: conda not found in PATH${NC}"
    echo "Using system Python instead..."
    PYTHON_CMD="python"
else
    echo -e "${GREEN}✅ Conda environment detected${NC}"
    PYTHON_CMD="conda run -n quant python"
fi

# Check if data exists
DATA_DIR="data/factors"
if [ ! -d "$DATA_DIR" ] || [ ! -f "$DATA_DIR/prices.parquet" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Data not found in $DATA_DIR${NC}"
    echo ""
    echo "Would you like to run the initial data backfill? (This may take 5-10 minutes)"
    echo "Press 'y' to backfill, or any other key to skip:"
    read -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Running backfill (this will take a few minutes)...${NC}"
        $PYTHON_CMD scripts/backfill_all.py --years 10
        echo -e "${GREEN}✅ Data backfill complete!${NC}"
    else
        echo -e "${YELLOW}Skipping backfill. You may see errors if data is missing.${NC}"
    fi
fi

# Check if streamlit is installed
echo ""
echo -e "${BLUE}Checking dependencies...${NC}"
if ! $PYTHON_CMD -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Streamlit not installed${NC}"
    echo "Installing streamlit..."
    
    if command -v conda &> /dev/null; then
        conda run -n quant pip install streamlit
    else
        pip install streamlit
    fi
    
    echo -e "${GREEN}✅ Streamlit installed${NC}"
else
    echo -e "${GREEN}✅ Streamlit is installed${NC}"
fi

# Launch the app
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}🚀 Launching Portfolio Simulator...${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

if command -v conda &> /dev/null; then
    conda run -n quant streamlit run apps/portfolio_simulator.py
else
    streamlit run apps/portfolio_simulator.py
fi

