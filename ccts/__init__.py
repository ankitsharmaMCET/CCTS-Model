"""CCTS (Carbon Capture Trading Simulation) package.

Provides agent-based model components for simulating
carbon markets, firm behavior, and analysis tools.
"""

from .model import CCTSModel
from .market import IndianCarbonMarket
from .agent import FirmAgent
from .analysis import analyze_results

__all__ = ["CCTSModel", "IndianCarbonMarket", "FirmAgent", "analyze_results"]
