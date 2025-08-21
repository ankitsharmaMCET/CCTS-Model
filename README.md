# Enhanced Carbon Credit Trading System (CCTS) Simulation

This project contains an agent-based model (ABM) to simulate the dynamics of the Indian Carbon Credit Trading System (CCTS). The model simulates the interactions between different firm agents and a central carbon market, allowing for the analysis of compliance rates, market stability, and economic impacts.

## Project Structure

The codebase is organized into a modular structure to ensure clarity, maintainability, and scalability.

-   **`ccts/`**: The main Python package for the simulation.
    -   `__init__.py`: Makes the directory a Python package.
    -   `market.py`: Defines the `EnhancedCarbonMarket` class, which manages trading, price dynamics, and regulatory compliance checks.
    -   `agent.py`: Defines the `FirmAgent` class, which represents individual firms with complex decision-making processes, including production optimization, abatement, and trading strategies.
    -   `model.py`: Defines the top-level `CCTSModel` class, which orchestrates the simulation, schedules agent actions, and collects data.
    -   `analysis.py`: Contains functions for performing post-simulation analysis on the collected data.
    -   `utils.py`: Holds utility functions and data classes used across the project for type safety and organization.
    -   `config.py`: Stores default configuration parameters for the simulation.
-   **`scripts/`**: Contains executable scripts, such as `run_simulation.py`, which serves as the entry point for running the simulation.
-   **`tests/`**: Contains unit tests for the main modules to ensure correctness and stability.
-   **`requirements.txt`**: Lists the necessary Python dependencies.
-   **`README.md`**: This file, providing an overview of the project.

## How to Run the Simulation

1.  **Clone the repository**:
    ```bash
    git clone [repository-url]
    cd ccts
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare input data**:
    The simulation requires an Excel file named `data/ccts_input_data.xlsx` with the following sheets:
    -   `Firms`: Contains data for each firm agent (e.g., `firm_id`, `sector`, `baseline_emissions`, `cash_on_hand`).
    -   `MACC`: Details of Marginal Abatement Cost Curve (MACC) projects available to each firm.
    -   `Config`: General simulation parameters like number of steps and initial carbon price.
    -   `Market_Params`: Parameters that control the behavior of the carbon market.

4.  **Run the simulation**:
    ```bash
    python scripts/run_simulation.py
    ```

The script will run the simulation, generate plots, and save a comprehensive results file named `enhanced_ccts_simulation_results.xlsx` in the root directory.

## Features

-   **Agent-Based Modeling**: Simulates individual firm behavior, including strategic decision-making on production, abatement, and trading.
-   **Dynamic Market Mechanism**: The `EnhancedCarbonMarket` module includes price discovery based on supply and demand, a market-maker for liquidity, and price caps/floors.
-   **Regulatory Compliance**: Agents are subject to compliance checks, including credit expiration and penalties for non-compliance, reflecting the Indian CCTS framework.
-   **Adaptive Learning**: Firms can learn from their past performance to adjust their market confidence and risk aversion, leading to more realistic and dynamic behavior.
-   **Comprehensive Data Collection**: The model uses MESA's `DataCollector` to track key metrics at both the model (e.g., carbon price, total emissions) and agent level (e.g., individual profit, emissions).
-   **Detailed Analysis & Visualization**: Post-simulation analysis functions provide insights into compliance rates, market performance, and economic impacts, accompanied by informative plots.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.