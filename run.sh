#!/bin/bash

# UnderwriteGPT Startup Script
# Easy way to run your app with different backends

echo "🛡️  UnderwriteGPT Launcher"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
if ! pip show streamlit &> /dev/null; then
    echo "❌ Packages not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Show menu
echo "Choose your LLM backend:"
echo ""
echo "1) GPT4All (Recommended for Mac - auto-downloads model)"
echo "2) Template Mode (Instant - no download needed)"
echo "3) HuggingFace (Advanced - larger download)"
echo "4) Ollama (If you have it installed)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "✅ Using GPT4All"
        export LLM_BACKEND=gpt4all
        ;;
    2)
        echo "✅ Using Template Mode (Fast!)"
        export LLM_BACKEND=template
        ;;
    3)
        echo "✅ Using HuggingFace"
        export LLM_BACKEND=huggingface
        ;;
    4)
        echo "✅ Using Ollama"
        export LLM_BACKEND=ollama
        ;;
    *)
        echo "⚠️  Invalid choice, using GPT4All"
        export LLM_BACKEND=gpt4all
        ;;
esac

echo ""
echo "🚀 Starting UnderwriteGPT..."
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Note: First run with GPT4All will download a 4GB model (5-10 min)"
echo "Press Ctrl+C to stop the app"
echo ""

# Run the app
streamlit run streamlit_app.py