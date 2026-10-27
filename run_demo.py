"""
Root launcher for running car racing visualizer & demo.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from car_racer.demo_racing import main

if __name__ == "__main__":
    main()
