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
# This is a critical fix for the UnicodeEncodeError
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
        # required_firm_columns = [
        #     'firm_id', 'sector', 'baseline_production', 'baseline_emissions_intensity',
        #     'target_emissions_intensity_2025', 'target_emissions_intensity_2026',
        #     'cash_on_hand', 'revenue_per_unit', 'is_covered'
        # ]

        required_firm_columns = [
            'firm_id', 'sector', 'baseline_production', 'baseline_emissions_intensity',
            'cash_on_hand', 'revenue_per_unit', 'is_covered'
        ]

        # Verify at least one target exists
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
    """CORRECTED: Load and process input data with more robust type handling"""
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
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                dfs[name] = df
                logger.info(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error loading {name} from {path}: {e}")
                raise

        if 'is_covered' in dfs['firms'].columns:
            bool_map = {'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False, True: True, False: False}
            dfs['firms']['is_covered'] = dfs['firms']['is_covered'].astype(str).str.lower().str.strip().replace(bool_map).astype(bool)

        if not validate_input_data(dfs['firms'], dfs['macc']):
            raise ValueError("Data validation failed")

        if not dfs['macc'].empty:
            macc_grouped = dfs['macc'].groupby('firm_id').apply(
                lambda group: group[['project_name', 'cost_per_tonne', 'abatement_potential']]
                .rename(columns={'project_name': 'name'})
                .to_dict('records'),
                include_groups=False
            ).to_dict()
            
            dfs['firms']['macc'] = dfs['firms']['firm_id'].map(macc_grouped).fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            dfs['firms']['macc'] = [[] for _ in range(len(dfs['firms']))]

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
                
                if (step + 1) % 6 == 0:
                    year = 2024 + ((step + 1) // 12)
                    month = ((step + 1) % 12)
                    if month == 0:
                        month = 12
                        year -= 1
                    
                    logger.info(f"Completed Step {step + 1} (Year {year}, Month {month}): "
                              f"Carbon Price ₹{model.market.carbon_price:.2f}, "
                              f"Market Volume: {model.market.daily_volume:.2f}")
                
            except Exception as e:
                logger.error(f"Error in simulation step {step + 1}: {e}")
                continue

        logger.info("Simulation completed successfully")
        return model

    except Exception as e:
        logger.error(f"Simulation initialization error: {e}")
        raise

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
    """Create comprehensive visualization plots"""
    try:
        output_path = Path(output_dir)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CCTS Simulation Results', fontsize=16, fontweight='bold')

        if not model_data.empty and 'Carbon_Price' in model_data.columns:
            ax1 = axes[0, 0]
            ax1.plot(model_data.index, model_data['Carbon_Price'], 'b-', linewidth=2, label='Carbon Price')
            ax1.set_xlabel('Simulation Step')
            ax1.set_ylabel('Carbon Price (₹)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('Carbon Price Evolution')
            ax1.grid(True, alpha=0.3)
            
            if 'Market_Volume' in model_data.columns:
                ax1_twin = ax1.twinx()
                ax1_twin.bar(model_data.index, model_data['Market_Volume'], 
                           alpha=0.3, color='orange', label='Daily Volume')
                ax1_twin.set_ylabel('Daily Volume', color='orange')
                ax1_twin.tick_params(axis='y', labelcolor='orange')
        
        if not model_data.empty:
            ax2 = axes[0, 1]
            if 'Total_Emissions' in model_data.columns:
                ax2.plot(model_data.index, model_data['Total_Emissions'], 'r-', 
                        linewidth=2, label='Total Emissions')
            if 'Total_Abated_Tons_Cumulative' in model_data.columns:
                ax2.plot(model_data.index, model_data['Total_Abated_Tons_Cumulative'], 'g-', 
                        linewidth=2, label='Total Abatement')
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Tonnes CO2e')
            ax2.set_title('Emissions vs Abatement')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if not model_data.empty and 'Compliance_Rate' in model_data.columns:
            ax3 = axes[1, 0]
            ax3.plot(model_data.index, model_data['Compliance_Rate'] * 100, 'purple', 
                    linewidth=2, marker='o', markersize=3)
            ax3.set_ylim(0, 105)
            ax3.set_xlabel('Simulation Step')
            ax3.set_ylabel('Compliance Rate (%)')
            ax3.set_title('Compliance Rate Over Time')
            ax3.grid(True, alpha=0.3)

        if not transactions_data.empty:
            ax4 = axes[1, 1]
            tx_summary = transactions_data.groupby('step').agg({
                'quantity': 'sum',
                'price': 'mean'
            })
            
            ax4.scatter(tx_summary['quantity'], tx_summary['price'], 
                       alpha=0.6, s=30, color='red')
            ax4.set_xlabel('Transaction Volume')
            ax4.set_ylabel('Average Price (₹)')
            ax4.set_title('Volume vs Price Relationship')
            ax4.grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Transaction Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Transaction Analysis')

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
        # Reset index to treat 'Step' as a column, fixing KeyError
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
        
        model = run_simulation(df_firms, market_params, config_params, num_steps)
        
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

