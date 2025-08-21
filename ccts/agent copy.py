

# =====================================================================
# ENHANCED FIRM AGENT (CORRECTED VERSION)
# =====================================================================

import logging
import random
import numpy as np
import traceback
from mesa import Agent
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from .utils import MarketOrder, ComplianceRecord, Transaction, safe_divide

logger = logging.getLogger(__name__)

class FirmAgent(Agent):
    """Enhanced Indian CCTS-compliant firm agent with advanced decision-making"""

    # Class-level constants for easy tuning
    CASH_BUFFER_RATIO = 0.2
    RISK_PREMIUM_MULTIPLIER = 0.05
    URGENCY_PREMIUM_CAP = 0.1
    RISK_DISCOUNT_MULTIPLIER = 0.03
    VOLUME_DISCOUNT_CAP = 0.08
    MAX_CREDITS_MULTIPLIER = 100
    PROFIT_HISTORY_MAX_LEN = 12
    DECISION_HISTORY_MAX_LEN = 12
    PRICE_FORECAST_MAX_LEN = 6
    PRODUCTION_STABILITY_PENALTY_FACTOR = 0.01
    MONTHS_PER_YEAR = 12

    def __init__(self, unique_id: int, model: 'CCTSModel', initial_data: Dict):
        super().__init__(unique_id, model)

        self._initialize_core_attributes(initial_data)
        self._initialize_compliance_attributes(initial_data)
        self._initialize_behavioral_attributes(initial_data)
        self._initialize_state_tracking()

        logger.debug(f"Firm {self.firm_id} initialized with baseline emissions: {self.baseline_emissions:.2f}")

    def _initialize_core_attributes(self, data: Dict) -> None:
        """Initialize core firm attributes with validation"""
        self.firm_id = data['firm_id']
        self.sector = data['sector']
        self.baseline_production = float(data['baseline_production'])
        self.baseline_emissions_intensity = float(data['baseline_emissions_intensity'])
        # self.target_emissions_intensity_2025 = float(data['target_emissions_intensity_2025'])
        # self.target_emissions_intensity_2026 = float(data['target_emissions_intensity_2026'])
        # Parse ALL target columns
        self.targets = {}
        for key, value in data.items():
            if key.startswith("target_emissions_intensity_"):
                try:
                    year = int(key.split("_")[-1])
                    self.targets[year] = float(value)
                except ValueError:
                    logger.warning(f"Invalid target year in {key}")

        # Initialize with earliest target
        self.target_years = sorted(self.targets.keys())
        self.current_target_year = self.target_years[0] if self.target_years else None



        self.macc = data.get('macc', [])
        self.risk_aversion = max(0.1, min(0.9, float(data.get('risk_aversion', 0.5))))
        self.cash = max(0.0, float(data['cash_on_hand']))
        self.revenue_per_unit = float(data['revenue_per_unit'])
        self.max_production_change_percent = float(data.get('max_production_change_percent', 0.2))
        self.variable_production_cost = float(data.get('variable_production_cost', 0))

        self.min_production = self.baseline_production * (1 - self.max_production_change_percent)
        self.max_production = self.baseline_production * (1 + self.max_production_change_percent)
        self.baseline_emissions = self.baseline_production * self.baseline_emissions_intensity

        if self.baseline_production <= 0 or self.revenue_per_unit <= 0:
            raise ValueError(f"Firm {self.firm_id}: Production and revenue must be positive")

    def _initialize_compliance_attributes(self, data: Dict) -> None:
        """Initialize CCTS compliance attributes"""
        self.is_covered_entity = bool(data.get('is_covered', False))
        self.coverage_threshold = self.model.coverage_threshold
        self.credit_vintages = {}
        self.penalty_history = []
        self.is_compliant = True
        self.verification_cost = self.model.market.verification_cost
        self.total_annual_emissions = 0.0
        self.total_annual_production = 0.0

    def _initialize_behavioral_attributes(self, data: Dict) -> None:
        """Initialize behavioral and learning attributes"""
        self.learning_rate = max(0.0, min(1.0, float(data.get('learning_rate', 0.1))))
        self.technology_readiness = max(0.0, min(1.0, float(data.get('technology_readiness', 0.5))))
        self.strategy_memory = []
        self.market_confidence = 0.5
        self.strategic_planning_horizon = 12
        self.adaptive_behavior = True

    def _initialize_state_tracking(self) -> None:
        """Initialize state tracking variables"""
        self.current_production = self.baseline_production
        self.current_emissions = self.baseline_emissions
        self.credits_owned = 0.0
        self.abated_tonnes_cumulative = 0.0
        self.abated_tonnes_this_step = 0.0
        self.completed_projects = []

        if self.target_years:
            self.current_target_year = min(self.target_years)
            self.current_target_intensity = self.targets[self.current_target_year]
        else:
            self.current_target_intensity = self.baseline_emissions_intensity

        #self.current_target_intensity = self.target_emissions_intensity_2025
        self.profit_history = []
        self.decision_history = []
        self.expected_price = self.model.market.carbon_price
        self.price_forecast = []
        
        self._last_credits_needed = 0.0
        self._last_credits_surplus = 0.0
        self._total_revenue = 0.0
        self._total_costs = 0.0

    def expire_credits(self, current_year: int) -> float:
        """Expire old credits and return the amount expired"""
        expired_credits = 0.0
        expired_vintages = []

        for vintage_year in list(self.credit_vintages.keys()):
            if current_year > vintage_year + self.model.market.credit_validity:
                expired_credits += self.credit_vintages[vintage_year]
                expired_vintages.append(vintage_year)

        for vintage in expired_vintages:
            del self.credit_vintages[vintage]

        self.credits_owned = max(0, self.credits_owned - expired_credits)

        if expired_credits > 0:
            logger.debug(f"Firm {self.firm_id}: {expired_credits} credits expired")

        return expired_credits

    def calculate_compliance_gap(self) -> float:
        """
        CORRECTED: Calculate compliance gap using cumulative annual data.
        A positive value means a deficit (needs to buy), a negative value means a surplus (can sell).
        """
        if not self.is_covered_entity:
            return 0.0
        
        allowed_emissions = self.current_target_intensity * self.total_annual_production
        gross_gap = self.total_annual_emissions - allowed_emissions
        
        return gross_gap

    def update_coverage_status(self) -> None:
        """Update coverage status based on sector and baseline emissions threshold"""
        self.is_covered_entity = (
            self.sector in self.model.covered_sectors and
            self.baseline_emissions >= self.coverage_threshold
        )

    def get_available_projects(self) -> List[Dict]:
        """Return list of abatement projects not yet completed"""
        completed_names = {p['name'] for p in self.completed_projects}
        return [proj for proj in self.macc if proj['name'] not in completed_names]

    def implement_abatement_portfolio(self, portfolio: List[Dict]) -> float:
        """Implement abatement projects if affordable and update state"""
        total_abatement = 0.0
        
        portfolio_sorted = sorted(portfolio, key=lambda x: x.get('cost_per_tonne', float('inf')))

        for project in portfolio_sorted:
            abatement_potential = project.get('abatement_potential', 0)
            cost_per_tonne = project.get('cost_per_tonne', 0)
            total_cost = abatement_potential * cost_per_tonne

            if self.cash >= total_cost:
                self.cash = max(0, self.cash - total_cost)
                self.abated_tonnes_cumulative += abatement_potential
                self.abated_tonnes_this_step += abatement_potential
                total_abatement += abatement_potential
                self.completed_projects.append(project.copy())
                logger.debug(
                    f"Firm {self.firm_id} implemented project {project['name']}: "
                    f"{abatement_potential} tonnes @ ₹{cost_per_tonne}/tonne"
                )
            else:
                logger.debug(f"Firm {self.firm_id} cannot afford project {project['name']}")
                break

        return total_abatement

    def update_price_expectations(self) -> None:
        """Update expected carbon price using recent market price history and adjustments"""
        try:
            price_history = self.model.market.price_history
            if len(price_history) > 3:
                weights = np.exp(np.linspace(-1, 0, min(5, len(price_history))))
                recent_prices = [p for p in price_history[-len(weights):] if p > 0]
                if recent_prices:
                    self.expected_price = np.average(recent_prices, weights=weights[-len(recent_prices):])
                else:
                    self.expected_price = self.model.market.carbon_price
            else:
                self.expected_price = max(1.0, self.model.market.carbon_price)

            risk_adjustment = (self.risk_aversion - 0.5) * 0.1
            self.expected_price *= (1 + risk_adjustment)

            volatility = getattr(self.model.market, 'volatility_index', 0.0)
            confidence_adjustment = (1 - volatility) * self.market_confidence * 0.05
            self.expected_price *= (1 + confidence_adjustment)

            self.expected_price = max(1.0, self.expected_price)

            self.price_forecast.append(self.expected_price)
            if len(self.price_forecast) > self.PRICE_FORECAST_MAX_LEN:
                self.price_forecast = self.price_forecast[-self.PRICE_FORECAST_MAX_LEN:]

        except Exception as e:
            logger.error(f"Price expectation update error for {self.firm_id}: {e}")
            self.expected_price = max(1.0, self.model.market.carbon_price)

    def select_optimal_abatement_portfolio(self) -> Tuple[List[Dict], float]:
        """Select abatement projects that yield positive NPV under cash constraints"""
        available_projects = self.get_available_projects()
        if not available_projects:
            return [], 0.0

        sorted_projects = sorted(available_projects, key=lambda x: x.get('cost_per_tonne', float('inf')))

        optimal_portfolio = []
        total_abatement = 0.0
        total_cost = 0.0

        for project in sorted_projects:
            abatement_potential = project.get('abatement_potential', 0)
            project_cost = abatement_potential * project.get('cost_per_tonne', float('inf'))

            if self._calculate_project_npv(project) > 0:
                if self.cash >= total_cost + project_cost:
                    optimal_portfolio.append(project)
                    total_cost += project_cost
                    total_abatement += abatement_potential
                else:
                    break

        portfolio_npv = (total_abatement * self.expected_price) - total_cost
        return optimal_portfolio, portfolio_npv

    def _calculate_project_npv(self, project: Dict) -> float:
        """Calculate Net Present Value of an abatement project"""
        try:
            cost_per_tonne = project.get('cost_per_tonne', float('inf'))
            abatement_potential = project.get('abatement_potential', 0)

            if cost_per_tonne == float('inf') or abatement_potential <= 0:
                return -float('inf')

            implementation_success_rate = self.technology_readiness * 0.8 + 0.2
            risk_adjusted_price = self.expected_price * implementation_success_rate

            benefits = abatement_potential * risk_adjusted_price
            costs = abatement_potential * cost_per_tonne

            return benefits - costs

        except Exception as e:
            logger.error(f"NPV calculation error for {self.firm_id}: {e}")
            return -float('inf')

    def optimize_production_decision(self) -> float:
        """CORRECTED: Optimize production level with better cash constraints"""
        try:
            max_affordable_cash = max(0, self.cash - (self.cash * self.CASH_BUFFER_RATIO))
            
            effective_intensity = safe_divide(
                max(0, self.baseline_emissions - self.abated_tonnes_cumulative), 
                self.baseline_production, 
                self.baseline_emissions_intensity
            )
            
            monthly_carbon_cost = effective_intensity * self.expected_price
            estimated_cost_per_unit = self.variable_production_cost + safe_divide(monthly_carbon_cost, self.MONTHS_PER_YEAR)
            
            if estimated_cost_per_unit > 0:
                constrained_max_production = min(
                    self.max_production, 
                    safe_divide(max_affordable_cash, estimated_cost_per_unit * self.MONTHS_PER_YEAR, self.max_production)
                )
            else:
                constrained_max_production = self.max_production

            constrained_max_production = max(self.min_production, constrained_max_production)

            def profit_function(production_array: np.ndarray) -> float:
                prod = production_array[0]
                
                revenue = prod * self.revenue_per_unit
                variable_costs = prod * self.variable_production_cost
                
                monthly_emissions = prod * effective_intensity
                
                monthly_allowed = safe_divide(
                    self.current_target_intensity * self.baseline_production, 
                    self.MONTHS_PER_YEAR
                )
                
                excess_emissions = max(0, monthly_emissions - monthly_allowed)
                carbon_costs = excess_emissions * self.expected_price
                
                profit = revenue - variable_costs - carbon_costs
                
                production_change = abs(prod - self.baseline_production) / self.baseline_production
                stability_penalty = production_change * self.risk_aversion * revenue * self.PRODUCTION_STABILITY_PENALTY_FACTOR
                
                return -(profit - stability_penalty)

            bounds = [(self.min_production, constrained_max_production)]
            initial_guess = [min(constrained_max_production, self.current_production)]

            result = minimize(
                profit_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={'ftol': 1e-6, 'maxiter': 100}
            )

            if result.success:
                optimized_production = result.x[0]
                return max(self.min_production, min(constrained_max_production, optimized_production))
            else:
                logger.warning(f"Production optimization failed for {self.firm_id}, using constrained baseline")
                return min(constrained_max_production, self.baseline_production)

        except Exception as e:
            logger.error(f"Production optimization error for {self.firm_id}: {e}")
            return self.baseline_production

    def calculate_trading_strategy(self) -> Tuple[float, float]:
        """CORRECTED: Calculate trading strategy based on compliance gap"""
        if not self.is_covered_entity:
            return 0.0, 0.0

        try:
            # Check compliance gap
            compliance_gap = self.calculate_compliance_gap()
            
            credits_needed = 0.0
            credits_surplus = 0.0

            if compliance_gap > 0:  # Has a deficit
                credits_needed = compliance_gap
                credits_surplus = 0.0
            elif compliance_gap < 0:  # Has a surplus
                credits_surplus = abs(compliance_gap)
                credits_needed = 0.0

            # Store for data collection
            self._last_credits_needed = credits_needed
            self._last_credits_surplus = credits_surplus
            
            return credits_needed, credits_surplus

        except Exception as e:
            logger.error(f"Trading strategy calculation error for {self.firm_id}: {e}")
            return 0.0, 0.0

    def place_market_orders(self, credits_needed: float, credits_surplus: float) -> None:
        """Place buy and sell orders in the market with improved pricing"""
        if not self.is_covered_entity:
            return

        min_cash_buffer = self.CASH_BUFFER_RATIO * self.cash

        try:
            # Place buy orders
            if credits_needed > 0:
                max_affordable = safe_divide(
                    max(0, self.cash - min_cash_buffer), 
                    self.expected_price
                )
                
                order_quantity = min(credits_needed, max_affordable)
                
                if order_quantity > 1e-6:
                    base_price = self.expected_price
                    risk_premium = self.risk_aversion * base_price * self.RISK_PREMIUM_MULTIPLIER
                    urgency_premium = min(order_quantity / self.model.market.market_maker_reserve, self.URGENCY_PREMIUM_CAP) * base_price

                    limit_price = base_price + risk_premium + urgency_premium
                    limit_price = max(self.model.market.min_price, min(self.model.market.max_price, limit_price))

                    self.model.market.place_order('buy', self, order_quantity, limit_price)
                else:
                    logger.warning(f"Firm {self.firm_id} has insufficient cash for any buy order ({self.cash:.2f})")

            # Place sell orders
            if credits_surplus > 0:
                order_quantity = min(credits_surplus, self.credits_owned)
                
                if order_quantity > 1e-6:
                    base_price = self.expected_price
                    risk_discount = self.risk_aversion * base_price * self.RISK_DISCOUNT_MULTIPLIER
                    volume_discount = min(order_quantity / self.model.market.market_maker_reserve, self.VOLUME_DISCOUNT_CAP) * base_price

                    limit_price = base_price - risk_discount - volume_discount
                    limit_price = max(self.model.market.min_price, min(self.model.market.max_price, limit_price))
                    
                    self.model.market.place_order('sell', self, order_quantity, limit_price)
                else:
                    logger.warning(f"Firm {self.firm_id} has no credits to sell despite surplus")

        except Exception as e:
            logger.error(f"Order placement error for {self.firm_id}: {e}")

    def learn_from_experience(self) -> None:
        """Adjust behavior based on recent performance"""
        if not self.adaptive_behavior or len(self.profit_history) < 3:
            return

        try:
            recent_profits = self.profit_history[-3:]
            profit_trend = np.mean(np.diff(recent_profits))

            if profit_trend > 0:
                self.market_confidence = min(1.0, self.market_confidence + 0.02)
                self.risk_aversion = max(0.1, self.risk_aversion - 0.01)
            else:
                self.market_confidence = max(0.0, self.market_confidence - 0.02)
                self.risk_aversion = min(0.9, self.risk_aversion + 0.01)

            if len(self.decision_history) >= 2:
                recent_decisions = self.decision_history[-2:]
                if all(d.get('npv_abate', 0) > 0 for d in recent_decisions):
                    self.technology_readiness = min(1.0, self.technology_readiness + 0.01)

        except Exception as e:
            logger.error(f"Learning error for {self.firm_id}: {e}")

    def add_credits_to_vintage(self, quantity: float, vintage_year: int) -> None:
        """Add credits to the appropriate vintage year"""
        if vintage_year not in self.credit_vintages:
            self.credit_vintages[vintage_year] = 0.0
        self.credit_vintages[vintage_year] += quantity
        self.credits_owned += quantity

    def step(self) -> None:
        """CORRECTED: Execute one simulation step"""
        try:
            # current_year = 2024 + (self.model.schedule.steps // self.MONTHS_PER_YEAR)
            
            # if self.model.schedule.steps % self.MONTHS_PER_YEAR == 0:
            #     self.total_annual_emissions = 0.0
            #     self.total_annual_production = 0.0
            #     self.abated_tonnes_this_step = 0.0
                
            #     if (self.model.schedule.steps // self.MONTHS_PER_YEAR) >= 1:
            #         self.current_target_intensity = self.target_emissions_intensity_2026
            #     else:
            #         self.current_target_intensity = self.target_emissions_intensity_2025

            current_simulation_year = 2024 + (self.model.schedule.steps // self.MONTHS_PER_YEAR)

            if self.model.schedule.steps % self.MONTHS_PER_YEAR == 0:
                # Reset annual counters
                self.total_annual_emissions = 0.0
                self.total_annual_production = 0.0
                self.abated_tonnes_this_step = 0.0
    
                # Update target for new year
                if current_simulation_year in self.targets:
                    self.current_target_year = current_simulation_year
                elif self.targets:
                    # Find latest target <= current year
                    past_targets = [y for y in self.target_years if y <= current_simulation_year]
                    self.current_target_year = max(past_targets) if past_targets else self.target_years[-1]
    
                self.current_target_intensity = self.targets.get(
                    self.current_target_year, 
                    self.baseline_emissions_intensity
                )




            self.update_price_expectations()

            portfolio, npv = self.select_optimal_abatement_portfolio()
            self.implement_abatement_portfolio(portfolio)
            
            # **CORRECTED:** Issue credits for abatement in this step
            if self.is_covered_entity and self.abated_tonnes_this_step > 0:
                #self.add_credits_to_vintage(self.abated_tonnes_this_step, current_year)
                self.add_credits_to_vintage(self.abated_tonnes_this_step, current_simulation_year)
                self.abated_tonnes_this_step = 0.0  # Reset for next step

            production = self.optimize_production_decision()
            self.current_production = production

            effective_intensity = safe_divide(
                max(0, self.baseline_emissions - self.abated_tonnes_cumulative),
                self.baseline_production,
                self.baseline_emissions_intensity
            )
            self.current_emissions = production * effective_intensity
            
            self.total_annual_emissions += self.current_emissions
            self.total_annual_production += self.current_production

            credits_needed, credits_surplus = self.calculate_trading_strategy()
            self.place_market_orders(credits_needed, credits_surplus)

            revenue = production * self.revenue_per_unit
            variable_costs = production * self.variable_production_cost
            
            monthly_allowed = safe_divide(
                self.current_target_intensity * self.baseline_production,
                self.MONTHS_PER_YEAR,
            )
            excess_emissions = max(0, self.current_emissions - monthly_allowed)
            carbon_costs = excess_emissions * self.expected_price
            
            monthly_profit = revenue - variable_costs - carbon_costs

            self.profit_history.append(monthly_profit)
            self.decision_history.append({
                'step': self.model.schedule.steps,
                'npv_abate': npv,
                'decision': 'optimize'
            })

            if len(self.profit_history) > self.PROFIT_HISTORY_MAX_LEN:
                self.profit_history = self.profit_history[-self.PROFIT_HISTORY_MAX_LEN:]
            if len(self.decision_history) > self.DECISION_HISTORY_MAX_LEN:
                self.decision_history = self.decision_history[-self.DECISION_HISTORY_MAX_LEN:]

            self.learn_from_experience()

            if self.model.schedule.steps % 12 == 0:
                logger.info(
                    f"Year {current_simulation_year} | Firm {self.firm_id}: "
                    f"Annual Prod: {self.total_annual_production:.2f}, "
                    f"Annual Emissions: {self.total_annual_emissions:.2f}, "
                    f"Compliance Gap: {self.calculate_compliance_gap():.2f}, "
                    f"Credits: {self.credits_owned:.2f}"
                )

        except Exception as e:
            logger.error(f"Step error for Firm {self.firm_id}: {e}")
            traceback.print_exc()