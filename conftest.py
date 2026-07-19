# Ensures the repo root is importable so `import copilot` resolves when
# running pytest from anywhere in the tree.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
