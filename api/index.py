import os
import sys

# Add the parent directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.api.main import app
