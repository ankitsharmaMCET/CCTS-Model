# =====================================================================
# ENHANCED CCTS MODEL - CORRECTED VERSION (with dynamic start year)
# =====================================================================

import logging
import pandas as pd
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from typing import Dict, List, Any
from .market import IndianCarbonMarket
from .agent import FirmAgent
from .utils import safe_divide

logger = logging.getLogger(__name__)


class CCTSModel(Model):
    """Enhanced CCTS Model with advanced features, monitoring, and reproducibility."""

    def __init__(self, firm_data: pd.DataFrame, market_params: Dict, model_config: Dict, seed: int = None):
        super().__init__()

        if seed is not None:
            self.random.seed(seed)
            np.random.seed(int(seed))

        # Get start year from config
        self.start_year = int(model_config.get("start_year", 2024))
        self.schedule = RandomActivation(self)
        self.coverage_threshold = float(model_config.get('coverage_threshold', 25000.0))
        self.model_config = model_config

        self.steps_per_year = int(model_config.get("steps_per_year", 12))

        initial_price = float(model_config.get('initial_carbon_price', 5000.0))
        self.market = IndianCarbonMarket(self, market_params, initial_price)

        self.covered_sectors = list(firm_data[firm_data['is_covered'] == True]['sector'].unique())

        self._create_agents(firm_data)
        self._setup_data_collection()
        self.yearly_summary_history = []

        self.model_stats = {
            'total_trades': 0,
            'total_volume': 0.0,
            'total_abatement': 0.0,
            'compliance_rate': 0.0
        }

        logger.info(
            f"CCTS Model initialized with {len(self.schedule.agents)} agents, "
            f"{len(self.covered_sectors)} covered sectors"
        )

    def _create_agents(self, firm_data: pd.DataFrame) -> None:
        """Create firm agents from data with better error handling"""
        firm_records = firm_data.to_dict('records')
        created_agents = 0

        for i, data in enumerate(firm_records):
            try:
                required_fields = [
                    'firm_id', 'sector', 'baseline_production', 'baseline_emissions_intensity',
                    'cash_on_hand', 'revenue_per_unit'
                ]
                
                missing_fields = [field for field in required_fields if field not in data or pd.isna(data[field])]
                if missing_fields:
                    logger.warning(f"Agent {i} missing required fields: {missing_fields}, skipping")
                    continue

                # Pass the model's start_year to the agent
                agent = FirmAgent(i, self, data)
                self.schedule.add(agent)
                created_agents += 1
                
            except Exception as e:
                logger.error(f"Error creating agent {i} (firm_id: {data.get('firm_id', 'unknown')}): {e}")
                continue

        logger.info(f"Created {created_agents} firm agents out of {len(firm_records)} records")

    def _setup_data_collection(self) -> None:
        """Setup comprehensive data collection with safe attribute access"""

        def safe_get_attr(agent, attr_name, default=0.0):
            """Safely get attribute value with default"""
            return getattr(agent, attr_name, default)

        def safe_get_list_last(agent, attr_name, default=0.0):
            """Safely get last item from a list attribute"""
            attr_list = getattr(agent, attr_name, [])
            return attr_list[-1] if attr_list else default

        model_reporters = {
            "Carbon_Price": lambda m: m.market.carbon_price,
            "Total_Emissions": lambda m: sum(safe_get_attr(a, 'current_emissions', 0) for a in m.schedule.agents),
            "Total_Abated_Tons_Step": lambda m: sum(safe_get_attr(a, 'abated_tonnes_this_step', 0) for a in m.schedule.agents),
            "Total_Abated_Tons_Cumulative": lambda m: sum(safe_get_attr(a, 'abated_tonnes_cumulative', 0) for a in m.schedule.agents),
            "Total_Production": lambda m: sum(safe_get_attr(a, 'current_production', 0) for a in m.schedule.agents),
            "Total_Annual_Emissions": lambda m: sum(safe_get_attr(a, 'total_annual_emissions', 0) for a in m.schedule.agents),
            "Total_Annual_Production": lambda m: sum(safe_get_attr(a, 'total_annual_production', 0) for a in m.schedule.agents),
            "Market_Volume": lambda m: m.market.daily_volume,
            "Price_Volatility": lambda m: getattr(m.market, "volatility_index", 0.0),
            "Compliance_Rate": lambda m: m._calculate_compliance_rate(),
            "Average_Profit": lambda m: m._calculate_average_profit(),
            "Total_Transactions": lambda m: len(m.market.transactions_log),
            "Market_Depth": lambda m: len(m.market.buy_orders) + len(m.market.sell_orders),
            "Buy_Orders": lambda m: len(m.market.buy_orders),
            "Sell_Orders": lambda m: len(m.market.sell_orders),
            "Price_Spread": lambda m: m.market.market_stats.get('average_spread', 0.0),
            "market_stability_reserve_fund": lambda m: m.market.market_stability_reserve_fund,
            "market_stability_reserve_credits": lambda m: m.market.market_stability_reserve_credits,
        }

        agent_reporters = {
            "Firm_ID": lambda a: safe_get_attr(a, 'firm_id', 'unknown'),
            "Sector": lambda a: safe_get_attr(a, 'sector', 'unknown'),
            "Emissions": lambda a: safe_get_attr(a, 'current_emissions', 0),
            "Current_Production": lambda a: safe_get_attr(a, 'current_production', 0),
            "Annual_Emissions": lambda a: safe_get_attr(a, 'total_annual_emissions', 0),
            "Annual_Production": lambda a: safe_get_attr(a, 'total_annual_production', 0),
            "Cash": lambda a: safe_get_attr(a, 'cash', 0),
            "Credits_Owned": lambda a: safe_get_attr(a, 'credits_owned', 0),
            "Abated_Tons_Cumulative": lambda a: safe_get_attr(a, 'abated_tonnes_cumulative', 0),
            "Is_Covered": lambda a: safe_get_attr(a, 'is_covered_entity', False),
            "Is_Compliant": lambda a: safe_get_attr(a, 'is_compliant', True),
            "Expected_Price": lambda a: safe_get_attr(a, 'expected_price', 0),
            "Risk_Aversion": lambda a: safe_get_attr(a, 'risk_aversion', 0.5),
            "Market_Confidence": lambda a: safe_get_attr(a, 'market_confidence', 0.5),
            "Current_Profit": lambda a: safe_get_list_last(a, 'profit_history', 0),
            "Compliance_Gap": lambda a: a.calculate_compliance_gap() if hasattr(a, 'calculate_compliance_gap') else 0,
            "Current_Target_Intensity": lambda a: safe_get_attr(a, 'current_target_intensity', 0),
            "Credits_Needed": lambda a: safe_get_attr(a, '_last_credits_needed', 0),
            "Credits_Surplus": lambda a: safe_get_attr(a, '_last_credits_surplus', 0),
            "Current_Target_Year": lambda a: safe_get_attr(a, 'current_target_year', 2025),
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

    def _calculate_compliance_rate(self) -> float:
        """Calculate current compliance rate safely"""
        try:
            covered_agents = [a for a in self.schedule.agents if getattr(a, 'is_covered_entity', False)]
            if not covered_agents:
                return 1.0

            compliant_agents = [a for a in covered_agents if getattr(a, 'is_compliant', True)]
            return safe_divide(len(compliant_agents), len(covered_agents), default=1.0)

        except Exception as e:
            logger.error(f"Compliance rate calculation error: {e}")
            return 0.0

    def _calculate_average_profit(self) -> float:
        """Calculate average profit across all agents"""
        try:
            profits = []
            for agent in self.schedule.agents:
                profit_history = getattr(agent, 'profit_history', [])
                if profit_history:
                    profits.append(profit_history[-1])
            
            return np.mean(profits) if profits else 0.0
        except Exception as e:
            logger.error(f"Average profit calculation error: {e}")
            return 0.0

    # def step(self) -> None:
    #     """Enhanced model step with proper Mesa scheduling"""
    #     try:
    #         self.schedule.step()
    #         self.market.step()
    #         self._update_model_statistics()
    #         self.datacollector.collect(self)
            
    #         if (self.schedule.steps) % self.steps_per_year == 0:
    #             self._log_annual_summary()

    #     except Exception as e:
    #         logger.error(f"Model step {self.schedule.steps} error: {e}")
    def step(self) -> None:
        """Enhanced model step with proper Mesa scheduling"""
        try:
            self.schedule.step()
            self.market.step()
            self._update_model_statistics()
            self.datacollector.collect(self)
        
            # CORRECTED: This condition now triggers at the end of the year (e.g., step 11, 23).
            # This ensures the annual summary is logged after all data for the year is complete.
            if (self.schedule.steps + 1) % self.steps_per_year == 0:
                self._log_annual_summary()

        except Exception as e:
            logger.error(f"Model step {self.schedule.steps} error: {e}")
    def _update_model_statistics(self) -> None:
        """Update comprehensive model statistics safely"""
        try:
            self.model_stats['total_trades'] = len(self.market.transactions_log)
            self.model_stats['total_volume'] = sum(t.quantity for t in self.market.transactions_log)
            self.model_stats['total_abatement'] = sum(getattr(a, 'abated_tonnes_cumulative', 0) for a in self.schedule.agents)
            self.model_stats['compliance_rate'] = self._calculate_compliance_rate()

        except Exception as e:
            logger.error(f"Model statistics update error: {e}")

    def _log_annual_summary(self) -> None:
        """Log annual summary statistics using data from the final step of the completed year."""
        try:
            year_end_step = self.schedule.steps
            year_start_step = year_end_step - self.steps_per_year + 1
            
            # Use the dynamic start_year attribute, and the calculation is correct.
            year = self.start_year + (year_end_step // self.steps_per_year)
            
            full_agent_data = self.datacollector.get_agent_vars_dataframe()
            
            if full_agent_data.index.get_level_values('Step').isin([year_end_step]).any():
                last_step_data = full_agent_data.xs(year_end_step, level='Step')
            else:
                logger.warning(f"No agent data found for step {year_end_step}, skipping annual summary.")
                return

            total_emissions = last_step_data['Annual_Emissions'].sum()
            total_production = last_step_data['Annual_Production'].sum()
            total_abatement = last_step_data['Abated_Tons_Cumulative'].sum()
            avg_intensity = safe_divide(total_emissions, total_production)

            annual_transactions = [
                t for t in self.market.transactions_log
                if year_start_step <= t.step <= year_end_step
            ]
            annual_volume = sum(t.quantity for t in annual_transactions)
            annual_trades = len(annual_transactions)

            annual_summary_data = {
        'year': year,
        'carbon_price': self.market.carbon_price,
        'total_annual_emissions': total_emissions,
        'total_annual_production': total_production,
        'total_abatement': total_abatement,
        'average_intensity': avg_intensity,
        'compliance_rate': self._calculate_compliance_rate(),
        'total_trades_this_year': annual_trades,
        'market_volume_this_year': annual_volume,
        'msr_fund_balance': self.market.market_stability_reserve_fund,
        'msr_credit_balance': self.market.market_stability_reserve_credits
    }

            self.yearly_summary_history.append(annual_summary_data)

            logger.info(f"=== YEAR {year} SUMMARY ===")
            logger.info(f"Carbon Price: ₹{self.market.carbon_price:.2f}")
            logger.info(f"Total Annual Emissions: {total_emissions:.2f} tonnes")
            logger.info(f"Total Annual Production: {total_production:.2f} units")
            logger.info(f"Total Abatement: {total_abatement:.2f} tonnes")
            logger.info(f"Average Intensity: {avg_intensity:.4f} tonnes/unit")
            logger.info(f"Compliance Rate: {self._calculate_compliance_rate():.1%}")
            logger.info(f"Total Trades: {annual_trades}")
            logger.info(f"Market Volume: {annual_volume:.2f} credits")
            logger.info(f"market_stability_reserve_fund: ₹{self.market.market_stability_reserve_fund:.2f}")
            logger.info(f"market_stability_reserve_credits: {self.market.market_stability_reserve_credits:.2f} credits")
            logger.info("=" * 30)

        except Exception as e:
            logger.error(f"Annual summary logging error: {e}")
    def get_model_summary(self) -> Dict:
        """Get a comprehensive model summary for analysis"""
        try:
            return {
                'steps_completed': self.schedule.steps,
                'years_completed': self.schedule.steps // self.steps_per_year,
                'final_carbon_price': self.market.carbon_price,
                'total_agents': len(self.schedule.agents),
                'covered_agents': len([a for a in self.schedule.agents if getattr(a, 'is_covered_entity', False)]),
                'model_stats': self.model_stats.copy(),
                'market_stats': self.market.market_stats.copy()
            }
        except Exception as e:
            logger.error(f"Model summary generation error: {e}")
            return {'error': str(e)}