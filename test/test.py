import os
import sys
import subprocess

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = os.environ.copy()
env["PYTHONPATH"] = project_root

subprocess.run(["python", "test/mnist.py"], env=env)