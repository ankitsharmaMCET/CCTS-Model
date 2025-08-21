# =====================================================================
# ENHANCED ANALYSIS FUNCTIONS (CORRECTED VERSION)
# =====================================================================


import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .model import CCTSModel
from .utils import safe_divide, safe_get_column

logger = logging.getLogger(__name__)

def analyze_results(model: CCTSModel, agent_df: pd.DataFrame,
                    transactions_df: pd.DataFrame, initial_firms_df: pd.DataFrame) -> Dict:
    """CORRECTED: Enhanced comprehensive results analysis"""
    try:
        if agent_df.empty:
            return {'error': 'No agent data available for analysis'}

        if 'Step' in agent_df.index.names:
            final_step = agent_df.index.get_level_values('Step').max()
            agent_df_reset = agent_df.reset_index()
        else:
            final_step = agent_df.index.levels[0].max()
            agent_df_reset = agent_df.reset_index().rename(columns={'level_0': 'Step'})
        
        merge_columns = ['firm_id'] if 'firm_id' in initial_firms_df.columns else ['Firm_ID']
        left_on = 'Firm_ID' if 'Firm_ID' in agent_df_reset.columns else 'AgentID'
        
        try:
            merged_agent_data = pd.merge(
                agent_df_reset,
                initial_firms_df,
                left_on=left_on,
                right_on=merge_columns[0],
                how='left',
                suffixes=('', '_initial')
            )
        except Exception as e:
            logger.warning(f"Merge failed, using agent data only: {e}")
            merged_agent_data = agent_df_reset.copy()

        final_agent_data = merged_agent_data[merged_agent_data['Step'] == final_step].copy()
        
        if final_agent_data.empty:
            return {'error': 'Final agent data is empty after filtering'}

        results = {}

        results.update(_analyze_compliance(final_agent_data, merged_agent_data, final_step))
        results.update(_analyze_market_performance(merged_agent_data, transactions_df))
        results.update(_analyze_sector_performance(final_agent_data))
        results.update(_analyze_abatement_effectiveness(final_agent_data))
        results.update(_analyze_economic_impacts(final_agent_data))
        results.update(_analyze_annual_summary(merged_agent_data, transactions_df))
        results.update(_analyze_model_performance(model))

        return results

    except Exception as e:
        logger.error(f"Results analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def _analyze_compliance(final_agent_data: pd.DataFrame, all_agent_data: pd.DataFrame, final_step: int) -> Dict:
    """CORRECTED: Analyze compliance performance with proper intensity calculations"""
    try:
        is_covered_col = safe_get_column(final_agent_data, 'Is_Covered', False)
        covered_entities = final_agent_data[is_covered_col == True].copy()

        if covered_entities.empty:
            return {'compliance_analysis': 'No covered entities found'}

        annual_emissions = safe_get_column(covered_entities, 'Annual_Emissions', 0)
        annual_production = safe_get_column(covered_entities, 'Annual_Production', 0)
        current_target_intensity = safe_get_column(covered_entities, 'Current_Target_Intensity', 0)
        credits_owned = safe_get_column(covered_entities, 'Credits_Owned', 0)
        
        allowed_emissions = current_target_intensity * annual_production
        
        gross_compliance_gap = annual_emissions - allowed_emissions
        
        net_compliance_gap = gross_compliance_gap - credits_owned
        net_compliance_gap = net_compliance_gap.clip(lower=0)
        
        is_compliant = net_compliance_gap <= 1e-6
        
        compliance_summary = pd.DataFrame({
            'Status': ['Compliant', 'Non-Compliant'],
            'Count': [is_compliant.sum(), (~is_compliant).sum()],
            'Percentage': [is_compliant.mean() * 100, (1 - is_compliant.mean()) * 100]
        })

        total_gross_gap = gross_compliance_gap.clip(lower=0).sum()
        total_net_gap = net_compliance_gap.sum()
        avg_gap = net_compliance_gap.mean()
        compliance_rate = is_compliant.mean()

        actual_intensity = safe_divide(annual_emissions.sum(), annual_production.sum())
        target_intensity_avg = current_target_intensity.mean()
        
        banking_analysis = None
        if 'Sector' in covered_entities.columns:
            banking_analysis = covered_entities.groupby('Sector').agg({
                'Credits_Owned': ['sum', 'mean', 'count']
            }).round(2)
            banking_analysis.columns = ['Total_Credits', 'Avg_Credits', 'Firm_Count']

        return {
            'compliance_summary': compliance_summary,
            'covered_entities_count': len(covered_entities),
            'total_gross_compliance_gap': float(total_gross_gap),
            'total_net_compliance_gap': float(total_net_gap),
            'average_compliance_gap': float(avg_gap),
            'overall_compliance_rate': float(compliance_rate),
            'actual_intensity': float(actual_intensity),
            'target_intensity_average': float(target_intensity_avg),
            'intensity_gap': float(actual_intensity - target_intensity_avg),
            'banking_analysis': banking_analysis.to_dict('index') if banking_analysis is not None else None
        }
    
    except Exception as e:
        logger.error(f"Compliance analysis error: {e}")
        return {'compliance_analysis_error': str(e)}

def _analyze_market_performance(agent_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict:
    """CORRECTED: Analyze carbon market performance"""
    try:
        results = {}

        if not transactions_df.empty and 'quantity' in transactions_df.columns and 'price' in transactions_df.columns:
            quantity = transactions_df['quantity']
            price = transactions_df['price']
            
            results.update({
                'total_trading_volume': float(quantity.sum()),
                'total_transaction_value': float((quantity * price).sum()),
                'total_transactions': int(len(transactions_df)),
                'average_trade_size': float(quantity.mean()),
                'average_trade_price': float(price.mean()),
                'min_trade_price': float(price.min()),
                'max_trade_price': float(price.max()),
                'median_trade_price': float(price.median()),
                'price_volatility': float(price.std() / price.mean()) if price.mean() > 0 else 0.0,
                'trade_size_distribution': quantity.describe().to_dict()
            })
        else:
            results.update({
                'total_trading_volume': 0.0,
                'total_transaction_value': 0.0,
                'total_transactions': 0,
                'message': 'No valid transaction data available'
            })

        if 'Step' in agent_df.columns:
            participation_data = []
            
            for step in sorted(agent_df['Step'].unique()):
                step_data = agent_df[agent_df['Step'] == step]
                
                credits_needed = safe_get_column(step_data, 'Credits_Needed', 0)
                credits_surplus = safe_get_column(step_data, 'Credits_Surplus', 0)
                
                buy_participation = (credits_needed > 0).mean()
                sell_participation = (credits_surplus > 0).mean()
                
                participation_data.append({
                    'step': step,
                    'buy_participation': buy_participation,
                    'sell_participation': sell_participation,
                    'total_participation': ((credits_needed > 0) | (credits_surplus > 0)).mean()
                })
            
            results['participation_analysis'] = participation_data

        return {'market_analysis': results}

    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        return {'market_analysis_error': str(e)}

def _analyze_sector_performance(final_agent_data: pd.DataFrame) -> Dict:
    """CORRECTED: Analyze performance by sector with safe column access"""
    try:
        sector_col = 'Sector' if 'Sector' in final_agent_data.columns else None
        if sector_col is None or final_agent_data[sector_col].isnull().all():
            return {'sector_analysis': 'No sector data available'}

        agg_columns = {}
        
        column_mappings = {
            'Emissions': ['Emissions', 'Current_Emissions', 'Annual_Emissions'],
            'Production': ['Current_Production', 'Production', 'Annual_Production'],
            'Abatement': ['Abated_Tons', 'Abatement', 'Total_Abatement'],
            'Credits': ['Credits_Owned', 'Credits', 'Total_Credits'],
            'Cash': ['Cash', 'Available_Cash', 'Current_Cash'],
            'Compliance': ['Is_Compliant', 'Compliant', 'Compliance_Status']
        }

        for metric, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in final_agent_data.columns:
                    agg_columns[metric] = col
                    break

        if not agg_columns:
            return {'sector_analysis_error': 'No suitable columns found for sector analysis'}

        sector_groups = final_agent_data.groupby(sector_col)
        sector_analysis = {}

        for sector, group in sector_groups:
            sector_data = {'firm_count': len(group)}
            
            for metric, col in agg_columns.items():
                values = group[col]
                
                if metric == 'Compliance':
                    if values.dtype == bool:
                        sector_data[f'{metric}_rate'] = values.mean()
                    else:
                        sector_data[f'{metric}_rate'] = (values != 0).mean()
                else:
                    sector_data.update({
                        f'{metric}_total': float(values.sum()),
                        f'{metric}_average': float(values.mean()),
                        f'{metric}_std': float(values.std())
                    })

            sector_analysis[sector] = sector_data

        return {'sector_analysis': sector_analysis}

    except Exception as e:
        logger.error(f"Sector analysis error: {e}")
        return {'sector_analysis_error': str(e)}

def _analyze_abatement_effectiveness(final_agent_data: pd.DataFrame) -> Dict:
    """CORRECTED: Analyze abatement project effectiveness"""
    try:
        results = {}

        abatement_col = None
        for col in ['Abated_Tons_Cumulative', 'Abated_Tons', 'Abatement', 'Total_Abatement']:
            if col in final_agent_data.columns:
                abatement_col = col
                break

        if abatement_col is None:
            return {'abatement_analysis': 'No abatement data available'}

        abatement_data = final_agent_data[abatement_col].fillna(0)

        results.update({
            'total_abatement': float(abatement_data.sum()),
            'average_abatement_per_firm': float(abatement_data.mean()),
            'median_abatement': float(abatement_data.median()),
            'firms_with_abatement': int((abatement_data > 0).sum()),
            'abatement_participation_rate': float((abatement_data > 0).mean()),
            'max_abatement': float(abatement_data.max()),
            'abatement_distribution': abatement_data.describe().to_dict()
        })

        if 'Sector' in final_agent_data.columns:
            sector_abatement = final_agent_data.groupby('Sector')[abatement_col].agg([
                'sum', 'mean', 'count'
            ]).round(2)
            sector_abatement.columns = ['Total', 'Average', 'Firms']
            results['abatement_by_sector'] = sector_abatement.to_dict('index')

        return {'abatement_analysis': results}

    except Exception as e:
        logger.error(f"Abatement analysis error: {e}")
        return {'abatement_analysis_error': str(e)}

def _analyze_economic_impacts(final_agent_data: pd.DataFrame) -> Dict:
    """CORRECTED: Analyze economic impacts with robust column handling"""
    try:
        results = {}

        economic_columns = {}
        column_mappings = {
            'profit': ['Current_Profit', 'Profit', 'Total_Profit'],
            'cash': ['Cash', 'Available_Cash', 'Current_Cash'],
            'revenue': ['Revenue', 'Total_Revenue', 'Annual_Revenue']
        }

        for metric, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in final_agent_data.columns:
                    economic_columns[metric] = col
                    break

        if not economic_columns:
            return {'economic_analysis': 'No economic data available'}

        for metric, column in economic_columns.items():
            data = final_agent_data[column].fillna(0)
            
            results.update({
                f'total_{metric}': float(data.sum()),
                f'average_{metric}': float(data.mean()),
                f'median_{metric}': float(data.median()),
                f'{metric}_distribution': data.describe().to_dict()
            })

        if 'cash' in economic_columns:
            cash_data = final_agent_data[economic_columns['cash']].fillna(0)
            results.update({
                'firms_with_positive_cash': int((cash_data > 0).sum()),
                'cash_positive_rate': float((cash_data > 0).mean()),
                'firms_in_financial_distress': int((cash_data <= 0).sum())
            })

        return {'economic_analysis': results}

    except Exception as e:
        logger.error(f"Economic analysis error: {e}")
        return {'economic_analysis_error': str(e)}

def _analyze_annual_summary(agent_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict:
    """CORRECTED: Annual summary with better data handling"""
    try:
        if agent_df.empty or 'Step' not in agent_df.columns:
            return {'annual_summary': 'No agent data for annual summary'}

        agent_df = agent_df.copy()
        agent_df['Year'] = (agent_df['Step'] // 12) + 1

        annual_data = []
        
        for year in sorted(agent_df['Year'].unique()):
            year_data = agent_df[agent_df['Year'] == year]
            
            summary = {'Year': year}
            
            metrics = {
                'Total_Production': ['Annual_Production', 'Total_Annual_Production'],
                'Total_Emissions': ['Annual_Emissions', 'Total_Annual_Emissions'],
                'Total_Abatement': ['Abated_Tons_Cumulative', 'Abated_Tons', 'Total_Abatement'],
                'Average_Profit': ['Current_Profit', 'Profit'],
                'Total_Credits': ['Credits_Owned', 'Credits']
            }

            for summary_name, possible_cols in metrics.items():
                value = 0.0
                for col in possible_cols:
                    if col in year_data.columns:
                        if 'Average' in summary_name:
                            value = float(year_data.groupby('AgentID')[col].last().mean())
                        else:
                            value = float(year_data.groupby('AgentID')[col].last().sum())
                        break
                summary[summary_name] = value

            annual_data.append(summary)

        if not transactions_df.empty and 'step' in transactions_df.columns:
            transactions_df = transactions_df.copy()
            transactions_df['Year'] = (transactions_df['step'] // 12) + 1
            
            transaction_summary = transactions_df.groupby('Year').agg({
                'quantity': 'sum',
                'price': 'mean'
            }).reset_index()
            transaction_summary.columns = ['Year', 'Total_Volume', 'Average_Price']
            
            annual_df = pd.DataFrame(annual_data)
            annual_df = pd.merge(annual_df, transaction_summary, on='Year', how='left')
            annual_df = annual_df.fillna(0)
            annual_data = annual_df.to_dict('records')

        return {'annual_summary': annual_data}

    except Exception as e:
        logger.error(f"Annual summary analysis error: {e}")
        return {'annual_summary_error': str(e)}

def _analyze_model_performance(model: CCTSModel) -> Dict:
    """Analyze overall model performance"""
    try:
        model_summary = model.get_model_summary()
        market_summary = model.market.get_market_summary()
        
        total_agents = len(model.schedule.agents)
        covered_agents = len([a for a in model.schedule.agents if getattr(a, 'is_covered_entity', False)])
        
        performance_metrics = {
            'simulation_steps_completed': model.schedule.steps,
            'simulation_years_completed': model.schedule.steps // model.steps_per_year,
            'total_agents': total_agents,
            'covered_entities': covered_agents,
            'coverage_rate': safe_divide(covered_agents, total_agents),
            'final_carbon_price': model.market.carbon_price,
            'price_volatility': model.market.volatility_index,
            'market_liquidity': len(model.market.buy_orders) + len(model.market.sell_orders),
            'total_transactions': len(model.market.transactions_log),
            'market_efficiency': safe_divide(len(model.market.transactions_log), model.schedule.steps)
        }
        
        return {
            'model_performance': performance_metrics,
            'model_summary': model_summary,
            'market_summary': market_summary
        }
        
    except Exception as e:
        logger.error(f"Model performance analysis error: {e}")
        return {'model_performance_error': str(e)}

def generate_summary_report(results: Dict) -> str:
    """Generate a human-readable summary report with enhancements."""
    try:
        report = []
        report.append("=" * 60)
        report.append("CCTS SIMULATION ANALYSIS REPORT")
        report.append("=" * 60)
        
        # New: Narrative Summary
        report.append("\nSIMULATION NARRATIVE SUMMARY:")
        final_price = results.get('model_performance', {}).get('final_carbon_price', 0)
        compliance_rate = results.get('overall_compliance_rate', 0) * 100
        total_abatement = results.get('abatement_analysis', {}).get('total_abatement', 0)
        total_transactions = results.get('market_analysis', {}).get('total_transactions', 0)

        narrative = (
            f"This simulation, running for {results.get('model_performance', {}).get('simulation_years_completed', 0)} years, "
            f"concluded with a final carbon price of ₹{final_price:.2f}. "
            f"The overall compliance rate was {compliance_rate:.1f}%, indicating a high degree of compliance with the CCTS regulations. "
            f"A total of {total_abatement:.2f} tonnes of CO2e were abated over the simulation period, "
            f"and the market saw {total_transactions} transactions."
        )
        report.append(narrative)

        if 'model_performance' in results:
            mp = results['model_performance']
            report.append(f"\nSIMULATION OVERVIEW:")
            report.append(f"- Years Completed: {mp.get('simulation_years_completed', 'N/A')}")
            report.append(f"- Total Agents: {mp.get('total_agents', 'N/A')}")
            report.append(f"- Covered Entities: {mp.get('covered_entities', 'N/A')}")
            report.append(f"- Final Carbon Price: ₹{mp.get('final_carbon_price', 0):.2f}")
        
        if 'overall_compliance_rate' in results:
            report.append(f"\nCOMPLIANCE PERFORMANCE:")
            report.append(f"- Overall Compliance Rate: {results['overall_compliance_rate']*100:.1f}%")
            report.append(f"- Total Compliance Gap: {results.get('total_net_compliance_gap', 0):.2f} tonnes")
            report.append(f"- Actual vs Target Intensity Gap: {results.get('intensity_gap', 0):.4f}")
        
        if 'market_analysis' in results:
            ma = results['market_analysis']
            report.append(f"\nMARKET PERFORMANCE:")
            report.append(f"- Total Trading Volume: {ma.get('total_trading_volume', 0):.2f} credits")
            report.append(f"- Total Transactions: {ma.get('total_transactions', 0)}")
            report.append(f"- Average Trade Price: ₹{ma.get('average_trade_price', 0):.2f}")
            report.append(f"- Price Volatility: {ma.get('price_volatility', 0)*100:.1f}%")
        
        if 'abatement_analysis' in results:
            aa = results['abatement_analysis']
            report.append(f"\nABATEMENT EFFECTIVENESS:")
            report.append(f"- Total Abatement: {aa.get('total_abatement', 0):.2f} tonnes")
            report.append(f"- Participation Rate: {aa.get('abatement_participation_rate', 0)*100:.1f}%")
            report.append(f"- Average per Firm: {aa.get('average_abatement_per_firm', 0):.2f} tonnes")
        
        if 'economic_analysis' in results:
            ea = results['economic_analysis']
            report.append(f"\nECONOMIC IMPACTS:")
            if 'total_profit' in ea:
                report.append(f"- Total Profit: ₹{ea['total_profit']:.2f}")
            if 'cash_positive_rate' in ea:
                report.append(f"- Firms with Positive Cash: {ea['cash_positive_rate']*100:.1f}%")

        # New: Per-Sector Breakdown
        if 'sector_analysis' in results and isinstance(results['sector_analysis'], dict):
            report.append("\nPER-SECTOR PERFORMANCE BREAKDOWN:")
            sector_data = results['sector_analysis']
            
            headers = ["Sector", "Firms", "Compliant (%)", "Abatement (t)", "Avg Profit (₹)"]
            report.append(f"{headers[0]:<15}{headers[1]:<10}{headers[2]:<20}{headers[3]:<20}{headers[4]:<15}")
            report.append("-" * 80)
            
            for sector, data in sector_data.items():
                firm_count = data.get('firm_count', 0)
                compliance_rate = data.get('Compliance_rate', 0) * 100
                total_abatement = data.get('Abatement_total', 0)
                avg_profit = data.get('Current_Profit_average', 0)
                
                row = (
                    f"{sector:<15}"
                    f"{firm_count:<10}"
                    f"{compliance_rate:<20.1f}"
                    f"{total_abatement:<20.2f}"
                    f"{avg_profit:<15.2f}"
                )
                report.append(row)

        report.append("=" * 60)
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Summary report generation error: {e}")
        return f"Error generating summary report: {str(e)}"