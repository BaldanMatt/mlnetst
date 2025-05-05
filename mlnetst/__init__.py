"""
mlnetst - Machine Learning Network Statistics Package
"""

from . import core
from . import utils
from . import plotter

__version__ = "0.1.0"
__all__ = ["core", "utils", "plotter"]

# Enable import shorthand 'mnst'
import sys

sys.modules['mnst'] = sys.modules[__name__]
