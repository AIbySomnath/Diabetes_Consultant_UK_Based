#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p .streamlit
mkdir -p data/rag

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please update the .env file with your API keys and configuration."
fi

echo "Setup complete!\n"
echo "To run the application locally:"
echo "1. Update the .env file with your API keys"
echo "2. Activate the virtual environment"
echo "3. Run: streamlit run app.py"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "1. Push this repository to GitHub"
echo "2. Go to https://share.streamlit.io/"
echo "3. Click 'New app' and follow the instructions"
