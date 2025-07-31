#!/usr/bin/env python3
"""
Setup script for LangGraph Studio configuration.
This script helps verify and set up the environment for LangGraph Studio.
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def check_langgraph_cli():
    """Check if LangGraph CLI is installed."""
    try:
        result = subprocess.run(
            ["langgraph", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ LangGraph CLI is installed: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ LangGraph CLI is not installed")
        return False


def install_langgraph_cli():
    """Install LangGraph CLI."""
    print("Installing LangGraph CLI...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "langgraph[cli]"],
            check=True
        )
        print("✅ LangGraph CLI installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install LangGraph CLI: {e}")
        return False


def check_langraph_json():
    """Check if langraph.json exists and is valid."""
    langraph_file = Path("langraph.json")
    if langraph_file.exists():
        print("✅ langraph.json file exists")
        return True
    else:
        print("❌ langraph.json file not found")
        return False


def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "SUPABASE_URL", 
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "DATABASE_URL",
        "STUDIO_TEST_USER_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "langgraph",
        "langchain", 
        "supabase",
        "pydantic",
        "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -e .")
        return False
    else:
        print("✅ All required packages are installed")
        return True


def main():
    """Main setup function."""
    print("🚀 LangGraph Studio Setup Check")
    print("=" * 40)
    
    # Check LangGraph CLI
    if not check_langgraph_cli():
        print("\nInstalling LangGraph CLI...")
        if not install_langgraph_cli():
            print("Failed to install LangGraph CLI. Please install manually:")
            print("pip install langgraph[cli]")
            return False
    
    print()
    
    # Check configuration files
    if not check_langraph_json():
        print("Please ensure langraph.json exists in the project root")
        return False
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies")
        return False
    
    print()
    
    # Check environment variables
    if not check_environment_variables():
        print("Please set up your environment variables")
        print("Copy env.example to .env and fill in your values")
        return False
    
    print()
    print("🎉 Setup complete! You can now run:")
    print("langgraph dev")
    print()
    print("This will start LangGraph Studio and open it in your browser.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 