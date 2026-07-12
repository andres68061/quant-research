#!/bin/bash
# Setup script for the quant project environment

echo "🚀 Setting up Quant Project Environment"
echo "========================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'quant'..."
conda create -n quant python=3.11 -y

# Activate environment
echo "🔄 Activating quant environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate quant

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter kernel
echo "🔧 Installing Jupyter kernel..."
python -m ipykernel install --user --name quant --display-name "Python (quant)"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/backups data/ml logs models results

# Set up environment variables
echo "🔑 Setting up environment variables..."
echo ""
echo "You can use a local .env file at the project root:"
echo "   cp .env.example .env   # then edit your values"
echo ""
echo "Or export variables in your shell (temporary):"
echo "   export FMP_API_KEY='your_fmp_key'"
echo "   export FRED_API_KEY='your_fred_key'"
echo "   export ALPHAVANTAGE_API_KEY='your_alpha_vantage_key'"
echo "   export OPENAI_API_KEY='your_openai_key'"
echo "   export BEA_API_KEY='your_bea_key'"
echo "   export REDDIT_CLIENT_ID='your_reddit_client_id'"
echo "   export REDDIT_CLIENT_SECRET='your_reddit_client_secret'"
echo "   export REDDIT_USER_AGENT='your_reddit_user_agent'"
echo "   export REDDIT_USERNAME='your_reddit_username'"
echo "   export REDDIT_PASSWORD='your_reddit_password'"
echo ""
echo "To persist across sessions (macOS zsh):"
echo "   echo 'export FMP_API_KEY=\"your_key_here\"' >> ~/.zshrc"
echo "   # Repeat for other variables as needed"
echo ""

# Test the environment
echo "🧪 Testing environment..."
python test_environment.py

# Test database system
echo "🗄️  Testing database system..."
python test_database_system.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Set your API keys as environment variables"
echo "   2. Open Cursor and select the quant environment"
echo "   3. Run: python example_usage.py"
echo "   4. Start Jupyter: jupyter lab"
echo ""
echo "🔧 To activate the environment manually:"
echo "   conda activate quant"
