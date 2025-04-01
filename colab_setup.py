import os
import subprocess
from IPython.display import clear_output

def setup_environment():
    """Setup the Colab environment with required dependencies"""
    print("Installing required packages...")
    
    # Install dependencies
    packages = [
        "transformers>=4.37.0",
        "torch>=2.0.0",
        "accelerate",
        "einops",
        "newsapi-python",
        "langchain-community",
        "langchain-huggingface",
        "faiss-cpu",
        "pytest"
    ]
    
    for package in packages:
        subprocess.run(f"pip install {package}", shell=True)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Setup environment variables
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""
NEWS_API_KEY=c4b96a9aeb6a4278ac728df4f4f265e8
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MAX_TOKENS=500
TEMPERATURE=0.7
LOG_LEVEL=INFO
""")
    
    clear_output()
    print("Setup completed! Don't forget to add your NewsAPI key to the .env file.")

if __name__ == "__main__":
    setup_environment()