#!/usr/bin/env python3
"""
Launcher script for the federated learning client.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the client
from clients.federated_client import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 