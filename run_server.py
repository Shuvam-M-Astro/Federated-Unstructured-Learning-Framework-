#!/usr/bin/env python3
"""
Launcher script for the federated learning server.
"""

import sys
import os
from pathlib import Path

print(">>> run_server.py is running")

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the server
from server.central_server import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 