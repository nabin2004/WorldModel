"""
Root launcher for training Mini World Models.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from car_racer.train import main

if __name__ == "__main__":
    main()
