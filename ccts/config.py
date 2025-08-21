"""
CCTS Simulation Configuration (CORRECTED VERSION)
Enhanced configuration with validation and comprehensive parameters.
"""

# =====================================================================
# CCTS SIMULATION CONFIGURATION
# =====================================================================

import logging
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import importlib.util
import json
import numpy as np

logger = logging.getLogger(__name__)

# =====================================================================
# SIMULATION PARAMETERS
# =====================================================================

SIMULATION_CONFIG = {
    "num_steps": 24,
    "steps_per_year": 12,
    "start_year": 2025,
    "coverage_threshold": 0000,
    "initial_carbon_price": 5000,
    
    "output_filename": 'enhanced_ccts_simulation_results.xlsx',
    "output_directory": 'output',
    "save_intermediate_results": False,
    "create_visualizations": True,
    
    "random_seed": None,
    "enable_learning": True,
    "enable_MarketStabilityReserve": True,
    "strict_compliance": True,
    
    "log_level": "INFO",
    "detailed_agent_logging": False,
    "save_agent_histories": True,
}

# =====================================================================
# MARKET PARAMETERS
# =====================================================================

MARKET_PARAMS = {
    "market_stability_reserve_credits": 10000,
    "market_stability_reserve_fund": 1000000,
    "MarketStabilityReserve_participation": True,
    "MarketStabilityReserve_spread": 0.02,
    
    "credit_validity": 5,
    "penalty_rate": 20000,
    "verification_cost": 50,
    "banking_allowed": True,
    "borrowing_allowed": False,
    
    "min_price": 3000,
    "max_price": 50000,
    "price_collar_active": True,
    "stability_reserve": 0.1,
    "circuit_breaker_threshold": 0.25,
    
    "price_sensitivity": 0.05,
    "base_volatility": 0.01,
    "max_daily_change": 0.15,
    "liquidity_spread": 0.02,
    "MarketStabilityReserve_impact": 0.5,
    
    "order_expiry_steps": 12,
    "partial_fill_allowed": True,
    "minimum_trade_size": 0.1,
    "maximum_order_size": 10000,
    
    "transaction_cost": 0,
    "market_impact_factor": 0.001,
    "clearing_frequency": 1,
}

# =====================================================================
# AGENT BEHAVIOR PARAMETERS
# =====================================================================

AGENT_BEHAVIOR_CONFIG = {
    "learning_enabled": True,
    "learning_rate_bounds": (0.01, 0.3),
    "memory_length": 12,
    "adaptation_speed": 0.02,
    
    "risk_aversion_bounds": (0.1, 0.9),
    "confidence_bounds": (0.0, 1.0),
    "planning_horizon": 12,
    
    "max_production_change": 0.2,
    "abatement_planning_horizon": 24,
    "technology_adoption_rate": 0.05,
    
    "cash_buffer_ratio": 0.2,
    "trading_aggressiveness": 0.1,
    "price_forecast_length": 6,
    
    "strategy_switching_threshold": 0.1,
    "collaboration_probability": 0.05,
    "information_sharing": False,


    "max_projects_per_year": 3,
    "abatement_budget_ratio": 0.3,
    "diminishing_returns_factor": 0.85,
    "scaling_penalty_factor": 0.015,
    "min_credit_reserve_ratio": 0.00,
    "project_implementation_variability": 0.3
}

# =====================================================================
# SECTOR-SPECIFIC PARAMETERS
# =====================================================================

SECTOR_CONFIG = {
    "covered_sectors": [
        "Power", "Steel", "Cement", "Aluminum", 
        "Chemicals", "Fertilizers", "Paper"
    ],
    
    "sector_characteristics": {
        "Power": {
            "baseline_intensity_range": (0.8, 1.2),
            "abatement_potential": 0.3,
            "technology_readiness": 0.7,
            "capital_intensity": "high"
        },
        "Steel": {
            "baseline_intensity_range": (1.8, 2.2),
            "abatement_potential": 0.2,
            "technology_readiness": 0.5,
            "capital_intensity": "very_high"
        },
        "Cement": {
            "baseline_intensity_range": (0.6, 0.9),
            "abatement_potential": 0.15,
            "technology_readiness": 0.4,
            "capital_intensity": "high"
        },
        "Aluminum": {
            "baseline_intensity_range": (1.5, 1.8),
            "abatement_potential": 0.25,
            "technology_readiness": 0.6,
            "capital_intensity": "very_high"
        },
        "Chemicals": {
            "baseline_intensity_range": (0.3, 0.7),
            "abatement_potential": 0.35,
            "technology_readiness": 0.8,
            "capital_intensity": "medium"
        },
        "Fertilizers": {
            "baseline_intensity_range": (0.4, 0.8),
            "abatement_potential": 0.3,
            "technology_readiness": 0.6,
            "capital_intensity": "medium"
        },
        "Paper": {
            "baseline_intensity_range": (0.2, 0.5),
            "abatement_potential": 0.4,
            "technology_readiness": 0.7,
            "capital_intensity": "medium"
        }
    }
}

# =====================================================================
# EMISSIONS AND TARGETS
# =====================================================================

EMISSIONS_CONFIG = {
    "national_intensity_target_2025": 0.8,
    "national_intensity_target_2026": 0.7,
    "annual_reduction_rate": 0.05,
    
    "baseline_year": 2020,
    "reference_intensity": 1.0,
    
    "reporting_frequency": 12,
    "verification_probability": 1.0,
    "measurement_uncertainty": 0.02,
    
    "offset_eligibility_threshold": 0.05,
    "credit_issuance_delay": 3,
    "additionality_test": True,
}

# =====================================================================
# VALIDATION FUNCTIONS
# =====================================================================

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters and set defaults for missing values.
    """
    validated = config.copy()
    
    numeric_validations = {
        'num_steps': (1, 1000, 24),
        'steps_per_year': (1, 365, 12),
        'coverage_threshold': (0, float('inf'), 25000),
        'initial_carbon_price': (1, float('inf'), 5000),
        'min_price': (1, float('inf'), 3000),
        'max_price': (1, float('inf'), 50000),
        'price_sensitivity': (0, 1, 0.05),
        'base_volatility': (0, 1, 0.01),
        'max_daily_change': (0, 1, 0.15),
    }
    
    for param, (min_val, max_val, default) in numeric_validations.items():
        if param in validated:
            try:
                value = float(validated[param])
                validated[param] = max(min_val, min(max_val, value))
            except (ValueError, TypeError):
                logger.warning(f"Invalid {param}: {validated[param]}, using default: {default}")
                validated[param] = default
        else:
            validated[param] = default
    
    if validated.get('max_price', 0) <= validated.get('min_price', 0):
        logger.warning("max_price must be greater than min_price, adjusting...")
        validated['max_price'] = validated['min_price'] * 10
    
    boolean_params = [
        'save_intermediate_results', 'create_visualizations', 'enable_learning',
        'enable_MarketStabilityReserve', 'strict_compliance', 'banking_allowed', 
        'borrowing_allowed', 'price_collar_active', 'partial_fill_allowed',
        'learning_enabled', 'information_sharing', 'additionality_test'
    ]
    
    for param in boolean_params:
        if param in validated:
            if isinstance(validated[param], str):
                validated[param] = validated[param].lower() in ['true', '1', 'yes']
            else:
                validated[param] = bool(validated[param])
    
    if 'log_level' in validated:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if validated['log_level'] not in valid_levels:
            validated['log_level'] = 'INFO'
    
    return validated

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with validation.
    Later configs override earlier ones.
    """
    merged = {}
    
    for config in configs:
        if isinstance(config, dict):
            merged.update(config)
        else:
            logger.warning(f"Skipping non-dict config: {type(config)}")
    
    return validate_config(merged)

def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from various file formats.
    Supports CSV, JSON, and Python files.
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"Config file not found: {filepath}, using defaults")
            return {}
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            if 'parameter' in df.columns and 'value' in df.columns:
                config = df.set_index('parameter')['value'].to_dict()
                for key, value in config.items():
                    try:
                        config[key] = float(value)
                    except (ValueError, TypeError):
                        if isinstance(value, str):
                             config[key] = value.lower() in ['true', '1', 'yes']
                        else:
                             pass
                return config
            else:
                logger.error("CSV config file must have 'parameter' and 'value' columns")
                return {}
                
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
                
        elif file_path.suffix.lower() == '.py':
            spec = importlib.util.spec_from_file_location("config_module", file_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            config = {}
            for name in dir(config_module):
                if name.isupper() and not name.startswith('_'):
                    config[name.lower()] = getattr(config_module, name)
            return config
            
        else:
            logger.error(f"Unsupported config file format: {file_path.suffix}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading config from {filepath}: {e}")
        return {}

def save_config_to_file(config: Dict[str, Any], filepath: str, format: str = 'csv') -> bool:
    """
    Save configuration to file in specified format.
    """
    try:
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df = pd.DataFrame([
                {'parameter': k, 'value': v, 'type': type(v).__name__}
                for k, v in config.items()
            ])
            df.to_csv(file_path, index=False)
            
        elif format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
                
        else:
            logger.error(f"Unsupported save format: {format}")
            return False
            
        logger.info(f"Configuration saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving config to {filepath}: {e}")
        return False

# =====================================================================
# SCENARIO CONFIGURATIONS
# =====================================================================

def get_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """
    Get predefined scenario configurations.
    """
    scenarios = {
        "baseline": {
            "description": "Baseline scenario with standard parameters",
            "price_sensitivity": 0.05,
            "penalty_rate": 20000,
            "MarketStabilityReserve_participation": True,
            "learning_enabled": True
        },
        "high_ambition": {
            "description": "High ambition scenario with stringent targets",
            "national_intensity_target_2025": 0.6,
            "national_intensity_target_2026": 0.5,
            "penalty_rate": 50000,
            "price_sensitivity": 0.08,
            "annual_reduction_rate": 0.08
        },
        "low_liquidity": {
            "description": "Low market liquidity scenario",
            "market_stability_reserve_credits": 500,
            "market_stability_reserve_fund": 500000,
            "liquidity_spread": 0.05,
            "price_sensitivity": 0.1,
            "max_daily_change": 0.25
        },
        "no_banking": {
            "description": "No credit banking allowed",
            "banking_allowed": False,
            "credit_validity": 1,
            "penalty_rate": 30000,
            "price_collar_active": True
        },
        "volatile_market": {
            "description": "Highly volatile market conditions",
            "base_volatility": 0.05,
            "max_daily_change": 0.3,
            "price_sensitivity": 0.15,
            "MarketStabilityReserve_impact": 0.3,
            "circuit_breaker_threshold": 0.4
        },
        "cooperative": {
            "description": "High cooperation between firms",
            "collaboration_probability": 0.2,
            "information_sharing": True,
            "learning_rate_bounds": (0.05, 0.4),
            "adaptation_speed": 0.05
        },
        "technology_push": {
            "description": "Rapid technology adoption scenario",
            "technology_adoption_rate": 0.1,
            "abatement_planning_horizon": 36,
            "offset_eligibility_threshold": 0.02,
            "additionality_test": False
        },
        "stress_test": {
            "description": "Market stress testing scenario",
            "initial_carbon_price": 15000,
            "penalty_rate": 80000,
            "coverage_threshold": 10000,
            "max_production_change": 0.4,
            "price_sensitivity": 0.2
        }
    }
    
    if scenario_name not in scenarios:
        logger.warning(f"Unknown scenario: {scenario_name}, returning baseline")
        scenario_name = "baseline"
    
    base_config = merge_configs(SIMULATION_CONFIG, MARKET_PARAMS, AGENT_BEHAVIOR_CONFIG)
    scenario_config = scenarios[scenario_name].copy()
    
    return merge_configs(base_config, scenario_config)

DEFAULT_CONFIG = merge_configs(
    SIMULATION_CONFIG,
    MARKET_PARAMS, 
    AGENT_BEHAVIOR_CONFIG,
    EMISSIONS_CONFIG
)

class CCTSConfigFactory:
    """
    Factory class for creating and managing CCTS configurations.
    """
    
    def __init__(self):
        self._base_configs = {
            'simulation': SIMULATION_CONFIG,
            'market': MARKET_PARAMS,
            'behavior': AGENT_BEHAVIOR_CONFIG,
            'sector': SECTOR_CONFIG,
            'emissions': EMISSIONS_CONFIG
        }
        
    def create_config(self, scenario: str = "baseline", 
                     config_files: List[str] = None,
                     overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a complete configuration by merging multiple sources.
        """
        config = get_scenario_config(scenario)
        
        if config_files:
            for file_path in config_files:
                file_config = load_config_from_file(file_path)
                config = merge_configs(config, file_config)
        
        if overrides:
            config = merge_configs(config, overrides)
        
        config = validate_config(config)
        
        logger.info(f"Created configuration for scenario: {scenario}")
        return config
    
    def get_base_config(self, config_type: str) -> Dict[str, Any]:
        """Get a base configuration by type."""
        return self._base_configs.get(config_type, {}).copy()
    
    def list_scenarios(self) -> List[str]:
        """List available predefined scenarios."""
        return list(get_scenario_config.__code__.co_names)

config_factory = CCTSConfigFactory()

__all__ = [
    'SIMULATION_CONFIG',
    'MARKET_PARAMS',
    'AGENT_BEHAVIOR_CONFIG', 
    'SECTOR_CONFIG',
    'EMISSIONS_CONFIG',
    'DEFAULT_CONFIG',
    'validate_config',
    'merge_configs',
    'load_config_from_file',
    'save_config_to_file',
    'get_scenario_config',
    'CCTSConfigFactory',
    'config_factory'
]