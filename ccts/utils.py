# =====================================================================
# UTILITY FUNCTIONS AND DATA CLASSES (CORRECTED VERSION)
# =====================================================================

import logging
from typing import Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    If conversion fails, return the provided default.
    """
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if abs(denominator) > 1e-9 else default

def safe_get_column(df: 'pd.DataFrame', column: str, default_value: Any = 0) -> 'pd.Series':
    """Safely get column from DataFrame with default values"""
    if column in df.columns:
        return df[column].fillna(default_value)
    else:
        logger.warning(f"Column '{column}' not found in DataFrame, using default value {default_value}")
        return pd.Series([default_value] * len(df), index=df.index)

@dataclass
class MarketOrder:
    """Represents a market order with all necessary information"""
    agent: 'FirmAgent'
    quantity: float
    timestamp: int
    order_type: str
    price: Optional[float] = None

@dataclass
class Transaction:
    """Represents a completed transaction"""
    step: int
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    transaction_type: str

@dataclass
class ComplianceRecord:
    """Tracks compliance status for covered entities"""
    year: int
    gap: float
    penalty: float
    is_compliant: bool