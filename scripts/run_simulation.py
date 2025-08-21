import pandas as pd
import logging
import traceback
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from pathlib import Path

# Import CCTS modules
from ccts.model import CCTSModel
from ccts.analysis import analyze_results, generate_summary_report
from ccts.utils import MarketOrder, Transaction, ComplianceRecord, safe_get_column, safe_divide

# Configure logging to handle Unicode characters (like ₹)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ccts_simulation.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def validate_input_data(df_firms: pd.DataFrame, df_macc: pd.DataFrame) -> bool:
    """Validate input data integrity"""
    try:
        required_firm_columns = [
            'firm_id', 'sector', 'baseline_production', 'baseline_emissions_intensity',
            'cash_on_hand', 'revenue_per_unit', 'is_covered'
        ]

        has_target = any(col.startswith("target_emissions_intensity_") 
                    for col in df_firms.columns)
        if not has_target:
            logger.error("Missing required target columns (prefix: target_emissions_intensity_)")
            return False
        
        missing_columns = [col for col in required_firm_columns if col not in df_firms.columns]
        if missing_columns:
            logger.error(f"Missing required columns in firms data: {missing_columns}")
            return False

        critical_positive = ['baseline_production', 'revenue_per_unit', 'cash_on_hand']
        for col in critical_positive:
            if col in df_firms.columns:
                invalid_count = (df_firms[col] <= 0).sum()
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} non-positive values in {col}")

        if 'is_covered' in df_firms.columns and not pd.api.types.is_bool_dtype(df_firms['is_covered']):
            logger.error("The 'is_covered' column is not of boolean type. Data loading failed.")
            return False

        if not df_macc.empty:
            required_macc_columns = ['firm_id', 'project_name', 'cost_per_tonne', 'abatement_potential']
            missing_macc = [col for col in required_macc_columns if col not in df_macc.columns]
            if missing_macc:
                logger.warning(f"Missing MACC columns: {missing_macc}")

        logger.info("Input data validation completed")
        return True

    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return False

def load_and_process_data(input_dir: str) -> tuple:
    """CORRECTED: Load and process input data with robust MACC handling"""
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        logger.info(f"Loading data from directory: {input_dir}")

        file_paths = {
            'firms': input_path / 'firms.csv',
            'macc': input_path / 'macc.csv', 
            'config': input_path / 'config.csv',
            'market_params': input_path / 'market_params.csv'
        }

        dfs = {}
        for name, path in file_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

            try:
                #df = pd.read_csv(path)
                # Locate this line in your existing code:
                #df = pd.read_csv(path)

                # Replace it with this corrected version:
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding='latin1')
                except Exception as e:
                    # Handle other potential errors gracefully
                    logger.error(f"Failed to load {name} with both UTF-8 and latin1 encodings: {e}")
                    raise
                
                df.columns = df.columns.str.strip()
                dfs[name] = df
                logger.info(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error loading {name} from {path}: {e}")
                raise

        if 'is_covered' in dfs['firms'].columns:
            bool_map = {'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False, True: True, False: False}
            dfs['firms']['is_covered'] = dfs['firms']['is_covered'].astype(str).str.lower().str.strip().map(bool_map).astype(bool)

        if not validate_input_data(dfs['firms'], dfs['macc']):
            raise ValueError("Data validation failed")

        # ROBUST MACC PROCESSING
        if not dfs['macc'].empty:
            # Create a copy to avoid modifying original
            macc_df = dfs['macc'].copy()
            
            # Rename project_name to name if exists
            if 'project_name' in macc_df.columns:
                macc_df.rename(columns={'project_name': 'name'}, inplace=True)
                
            # Ensure required columns exist
            required_columns = ['name', 'cost_per_tonne', 'abatement_potential']
            for col in required_columns:
                if col not in macc_df.columns:
                    raise ValueError(f"Missing required column in MACC data: {col}")
            
            # Add optional columns if missing
            if 'implementation_time' not in macc_df.columns:
                macc_df['implementation_time'] = 6  # Default value
                
            if 'type' not in macc_df.columns:
                macc_df['type'] = 'general'  # Default value
                
            # Group by firm_id
            macc_grouped = macc_df.groupby('firm_id').apply(
                lambda group: group[['name', 'cost_per_tonne', 'abatement_potential', 'implementation_time', 'type']]
                .to_dict('records'),
                include_groups=False
            ).to_dict()
            
            dfs['firms']['macc'] = dfs['firms']['firm_id'].map(macc_grouped).fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            dfs['firms']['macc'] = [[] for _ in range(len(dfs['firms']))]

        # Rest of the processing remains the same...
        config_params = dfs['config'].set_index('parameter')['value'].to_dict()
        market_params = dfs['market_params'].set_index('parameter')['value'].to_dict()

        numeric_config = {}
        for key, value in config_params.items():
            try:
                numeric_config[key] = float(value)
            except (ValueError, TypeError):
                numeric_config[key] = value

        numeric_market = {}  
        for key, value in market_params.items():
            try:
                numeric_market[key] = float(value)
            except (ValueError, TypeError):
                numeric_market[key] = value

        logger.info("Data loading and processing completed successfully")
        return dfs['firms'], dfs['macc'], numeric_config, numeric_market

    except Exception as e:
        logger.error(f"Data loading error: {e}")
        raise

# ... (rest of the file remains unchanged) ...

def run_simulation(df_firms: pd.DataFrame, market_params: dict, config_params: dict, 
                  num_steps: int) -> CCTSModel:
    """Run the CCTS simulation with progress tracking"""
    try:
        logger.info(f"Initializing CCTS model with {len(df_firms)} firms")
        
        seed = config_params.get('random_seed')
        model = CCTSModel(df_firms, market_params, config_params, seed=seed)
        
        logger.info(f"Starting simulation for {num_steps} steps ({num_steps/12:.1f} years)")
        
        for step in range(num_steps):
            try:
                model.step()
                
                # if (step + 1) % 6 == 0:
                #     # Corrected year and month calculation
                #     current_year = model.start_year + (step // 12)
                #     current_month = (step % 12) + 1
                    
                if (step + 1) % (model.steps_per_year // 2) == 0:  # Log twice a year
                    current_year = model.start_year + (step // model.steps_per_year)
                    current_month = (step % model.steps_per_year) + 1

                    logger.info(f"Completed Step {step + 1} (Year {current_year}, Month {current_month}): "
                              f"Carbon Price ₹{model.market.carbon_price:.2f}, "
                              f"Market Volume: {model.market.daily_volume:.2f}")
                
            except Exception as e:
                logger.error(f"Error in simulation step {step + 1}: {e}")
                continue

        logger.info("Simulation completed successfully")
        return model

    except Exception as e:
        logger.error(f"SIMULATION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

# ... (rest of the file remains unchanged) ...
# ... (rest of the file remains unchanged) ...
def save_results(model: CCTSModel, analysis_results: dict, output_dir: str,
                df_firms: pd.DataFrame, df_macc: pd.DataFrame, 
                config_params: dict, market_params: dict) -> None:
    """Save simulation results with comprehensive error handling"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {output_path}")

        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()
        
        transactions_data = pd.DataFrame([
            {
                'step': t.step,
                'buyer_id': t.buyer_id, 
                'seller_id': t.seller_id,
                'quantity': t.quantity,
                'price': t.price,
                'transaction_type': t.transaction_type
            }
            for t in model.market.transactions_log
        ])

    # New: Save the yearly summary history
        try:
            yearly_summary_df = pd.DataFrame(model.yearly_summary_history)
            if not yearly_summary_df.empty:
                yearly_summary_df.to_csv(output_path / 'yearly_summary_data.csv', index=False)
                logger.info("Saved yearly summary data.")
            else:
                logger.warning("Yearly summary data is empty, skipping save.")
        except Exception as e:
            logger.error(f"Error saving yearly summary data: {e}")    

# In run_simulation.py within the save_results function
# ... (after the transactions_data DataFrame is created)

# ADDITION: Create a DataFrame for agent-level compliance gap history
        compliance_gap_records = []
        for agent in model.schedule.agents:
            if hasattr(agent, 'compliance_gap_history'):
                for i, gap in enumerate(agent.compliance_gap_history):
                        compliance_gap_records.append({
                        'Firm_ID': agent.firm_id,
                        'Step': i,
                        'Compliance_Gap': gap
                    })

        # Save the compliance gap history DataFrame
        if compliance_gap_records:
            compliance_gap_df = pd.DataFrame(compliance_gap_records)
            compliance_gap_df.to_csv(output_path / 'agent_compliance_gap_history.csv', index=False)
            logger.info("Saved agent compliance gap history.")
        else:
            logger.warning("Agent compliance gap history is empty, skipping save.")


        agent_financial_history = []
        for agent in model.schedule.agents:
            # Check if the agent has the new history attributes
            if hasattr(agent, 'revenue_history'):
                # Find the minimum length of all history lists to prevent IndexError
                min_len = min(
                    len(agent.revenue_history),
                    len(agent.variable_cost_history),
                    len(agent.carbon_cost_history),
                    len(agent.abatement_cost_history),
                    len(agent.profit_history)
                )

                for i in range(min_len):
                    agent_financial_history.append({
                        'Firm_ID': agent.firm_id,
                        'Step': i,
                        'Revenue': agent.revenue_history[i],
                        'Variable_Costs': agent.variable_cost_history[i],
                        'Carbon_Costs': agent.carbon_cost_history[i],
                        'Abatement_Costs': agent.abatement_cost_history[i],
                        'Profit': agent.profit_history[i]
                    })

        # Save the financial history DataFrame
        if agent_financial_history:
            financial_df = pd.DataFrame(agent_financial_history)
            financial_df.to_csv(output_path / 'agent_financial_history.csv', index=False)
            logger.info("Saved agent financial history.")
        else:
            logger.warning("Agent financial history is empty, skipping save.")


        files_to_save = {
            'model_data.csv': model_data,
            'agent_data.csv': agent_data,
            'transactions_log.csv': transactions_data,
            'initial_firms_data.csv': df_firms,
            'initial_macc_data.csv': df_macc
        }

        for filename, data in files_to_save.items():
            try:
                if not data.empty:
                    data.to_csv(output_path / filename, index=True)
                    logger.info(f"Saved {filename}: {len(data)} rows")
                else:
                    logger.warning(f"Empty dataframe for {filename}")
            except Exception as e:
                logger.error(f"Error saving {filename}: {e}")

        # ADDITION: Normalize and save order book history
        try:
            order_book_records = []
            for entry in model.market.order_book_history:
                step = entry['step']
                for price, quantity in entry['buy_side'].items():
                    order_book_records.append({'step': step, 'price': price, 'quantity': quantity, 'side': 'buy'})
                for price, quantity in entry['sell_side'].items():
                    order_book_records.append({'step': step, 'price': price, 'quantity': quantity, 'side': 'sell'})

            if order_book_records:
                order_book_data = pd.DataFrame(order_book_records)
                order_book_data.to_csv(output_path / 'order_book_history.csv', index=False)
                logger.info(f"Saved normalized order book history: {len(order_book_data)} records")
            else:
                logger.warning("Order book history is empty, skipping save.")
        except Exception as e:
            logger.error(f"Error saving order book history: {e}")

        try:
            config_df = pd.DataFrame([
                {'parameter': k, 'value': v, 'type': 'config'} 
                for k, v in config_params.items()
            ] + [
                {'parameter': k, 'value': v, 'type': 'market'} 
                for k, v in market_params.items()
            ])
            config_df.to_csv(output_path / 'simulation_config.csv', index=False)
            logger.info("Saved simulation configuration")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

        try:
            for key, value in analysis_results.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(output_path / f'{key}.csv', index=True)
                elif isinstance(value, dict) and key.endswith('_analysis'):
                    try:
                        df = pd.DataFrame.from_dict(value, orient='index')
                        df.to_csv(output_path / f'{key}.csv')
                    except:
                        pd.Series(value).to_csv(output_path / f'{key}.csv')

            analysis_summary = {k: v for k, v in analysis_results.items() 
                              if not isinstance(v, (pd.DataFrame, dict, list))}
            if analysis_summary:
                summary_df = pd.DataFrame.from_dict(analysis_summary, orient='index', columns=['Value'])
                summary_df.index.name = 'Metric'
                summary_df.to_csv(output_path / 'analysis_summary.csv')

            logger.info("Saved analysis results")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")

        try:
            summary_report = generate_summary_report(analysis_results)
            with open(output_path / 'simulation_summary_report.txt', 'w', encoding='utf-8') as f:
                f.write(summary_report)
            logger.info("Saved summary report")
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")

    except Exception as e:
        logger.error(f"Error in save_results: {e}")
        raise

def create_visualizations(model_data: pd.DataFrame, agent_data: pd.DataFrame, 
                          transactions_data: pd.DataFrame, output_dir: str) -> None:
    """Create comprehensive visualization plots with enhancements."""
    try:
        output_path = Path(output_dir)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CCTS Simulation Results', fontsize=16, fontweight='bold')

        # --- Enhanced Carbon Price Plot (ax1) ---
        if not model_data.empty and 'Carbon_Price' in model_data.columns:
            ax1 = axes[0, 0]
            ax1.plot(model_data.index, model_data['Carbon_Price'], 'b-', linewidth=2, label='Carbon Price')
            
            # Add a rolling average price line
            if len(model_data) > 12:
                rolling_avg = model_data['Carbon_Price'].rolling(window=12, min_periods=1).mean()
                ax1.plot(model_data.index, rolling_avg, 'r--', linewidth=2, label='12-Step Rolling Average')
            
            ax1.set_xlabel('Simulation Step')
            ax1.set_ylabel('Carbon Price (₹)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('Carbon Price Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            if 'Market_Volume' in model_data.columns:
                ax1_twin = ax1.twinx()
                ax1_twin.bar(model_data.index, model_data['Market_Volume'], 
                            alpha=0.3, color='orange', label='Daily Volume')
                ax1_twin.set_ylabel('Daily Volume', color='orange')
                ax1_twin.tick_params(axis='y', labelcolor='orange')
                ax1_twin.legend(loc='upper right')
        
        # --- Stacked Area Chart for Emissions (ax2) ---
        ax2 = axes[0, 1]
        if not agent_data.empty and 'Sector' in agent_data.columns and 'Annual_Emissions' in agent_data.columns:
            # Pivot the agent data to get emissions per sector per step
            emissions_by_sector = agent_data.reset_index().pivot_table(
                index='Step', columns='Sector', values='Annual_Emissions', aggfunc='sum'
            ).fillna(0)

            emissions_by_sector.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Tonnes CO2e')
            ax2.set_title('Annual Emissions by Sector')
            ax2.legend(title='Sector', loc='upper left')
            ax2.grid(True, alpha=0.3)
        else:
            if not model_data.empty and 'Total_Emissions' in model_data.columns:
                ax2.plot(model_data.index, model_data['Total_Emissions'], 'r-', linewidth=2, label='Total Emissions')
                if 'Total_Abated_Tons_Cumulative' in model_data.columns:
                    ax2.plot(model_data.index, model_data['Total_Abated_Tons_Cumulative'], 'g-', linewidth=2, label='Total Abatement')
                ax2.set_xlabel('Simulation Step')
                ax2.set_ylabel('Tonnes CO2e')
                ax2.set_title('Emissions vs Abatement')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # --- Compliance Rate Plot (ax3) ---
        if not model_data.empty and 'Compliance_Rate' in model_data.columns:
            ax3 = axes[1, 0]
            ax3.plot(model_data.index, model_data['Compliance_Rate'] * 100, 'purple', 
                     linewidth=2, marker='o', markersize=3)
            ax3.set_ylim(0, 105)
            ax3.set_xlabel('Simulation Step')
            ax3.set_ylabel('Compliance Rate (%)')
            ax3.set_title('Compliance Rate Over Time')
            ax3.grid(True, alpha=0.3)
        
        # --- Bid-Ask Spread Plot (ax4) ---
        ax4 = axes[1, 1]
        if not model_data.empty and 'Price_Spread' in model_data.columns:
            ax4.plot(model_data.index, model_data['Price_Spread'] * 100, 'g-', linewidth=2)
            ax4.set_xlabel('Simulation Step')
            ax4.set_ylabel('Bid-Ask Spread (%)')
            ax4.set_title('Market Bid-Ask Spread Over Time')
            ax4.grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Spread Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Market Spread Analysis')

        plt.tight_layout()
        
        plot_path = output_path / 'simulation_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {plot_path}")

        if not agent_data.empty and 'Sector' in agent_data.columns:
            create_sector_analysis_plot(agent_data, output_path)

    except Exception as e:
        logger.error(f"Visualization error: {e}")

def create_sector_analysis_plot(agent_data: pd.DataFrame, output_path: Path) -> None:
    """Create sector-specific analysis plots"""
    try:
        agent_data_flat = agent_data.reset_index()
        final_step = agent_data_flat['Step'].max()
        final_data = agent_data_flat[agent_data_flat['Step'] == final_step]
        
        if 'Sector' not in final_data.columns:
            return
            
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 2, 1)
        if 'Annual_Emissions' in final_data.columns:
            emissions_by_sector = final_data.groupby('Sector')['Annual_Emissions'].sum()
            emissions_by_sector.plot(kind='bar', color='lightcoral')
            plt.title('Annual Emissions by Sector')
            plt.ylabel('Tonnes CO2e')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        if 'Abated_Tons_Cumulative' in final_data.columns:
            abatement_by_sector = final_data.groupby('Sector')['Abated_Tons_Cumulative'].sum()
            abatement_by_sector.plot(kind='bar', color='lightgreen')
            plt.title('Total Abatement by Sector')
            plt.ylabel('Tonnes CO2e')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        if 'Credits_Owned' in final_data.columns:
            credits_by_sector = final_data.groupby('Sector')['Credits_Owned'].sum()
            credits_by_sector.plot(kind='bar', color='lightblue')
            plt.title('Credits Owned by Sector')
            plt.ylabel('Credits')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        if 'Is_Compliant' in final_data.columns:
            compliance_by_sector = final_data.groupby('Sector')['Is_Compliant'].mean()
            compliance_by_sector.plot(kind='bar', color='gold')
            plt.title('Compliance Rate by Sector')
            plt.ylabel('Compliance Rate')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'sector_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved sector analysis plot")
        
    except Exception as e:
        logger.error(f"Sector plot error: {e}")

def main():
    """CORRECTED: Enhanced main execution function"""
    try:
        if len(sys.argv) > 1:
            input_dir = sys.argv[1]
        else:
            input_dir = 'input'
            
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        else:
            output_dir = 'output'

        logger.info("=" * 60)
        logger.info("STARTING ENHANCED CCTS CARBON TRADING SIMULATION")
        logger.info("=" * 60)
        
        df_firms, df_macc, config_params, market_params = load_and_process_data(input_dir)
        
        num_steps = int(config_params.get('num_steps', 24))
        logger.info(f"Configuration: {num_steps} steps ({num_steps/12:.1f} years)")
        
        model = CCTSModel(df_firms, market_params, config_params, seed=config_params.get('random_seed'))
        
        logger.info(f"Starting simulation for {num_steps} steps ({num_steps/12:.1f} years)")
        
        # New, corrected code
        for step in range(num_steps):
            try:
                model.step()
        
                if (step + 1) % 6 == 0:
                    current_year = model.start_year + (step // model.steps_per_year)
                    current_month = (step % model.steps_per_year) + 1
            
                    logger.info(f"Completed Step {step + 1} (Year {current_year}, Month {current_month}): "
                            f"Carbon Price ₹{model.market.carbon_price:.2f}, "
                            f"Market Volume: {model.market.daily_volume:.2f}")
            except Exception as e:
                logger.error(f"Error in simulation step {step + 1}: {e}")
                continue

        logger.info("Simulation completed successfully")
        
        logger.info("Collecting and analyzing simulation results...")
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()
        transactions_data = pd.DataFrame([vars(t) for t in model.market.transactions_log])
        
        analysis_results = analyze_results(model, agent_data, transactions_data, df_firms)
        
        logger.info("Creating visualizations...")
        create_visualizations(model_data, agent_data, transactions_data, output_dir)
        
        logger.info("Saving results...")
        save_results(model, analysis_results, output_dir, df_firms, df_macc, 
                    config_params, market_params)
        
        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if 'model_performance' in analysis_results:
            mp = analysis_results['model_performance']
            logger.info(f"Final Carbon Price: ₹{mp.get('final_carbon_price', 0):.2f}")
            logger.info(f"Total Transactions: {mp.get('total_transactions', 0)}")
            logger.info(f"Market Efficiency: {mp.get('market_efficiency', 0):.2f} transactions/step")
        
        if 'overall_compliance_rate' in analysis_results:
            logger.info(f"Final Compliance Rate: {analysis_results['overall_compliance_rate']*100:.1f}%")
        
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"SIMULATION FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()