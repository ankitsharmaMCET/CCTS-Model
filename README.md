# Enhanced Carbon Credit Trading System (CCTS) — Agent-Based Simulation

[![Status: Active Development](https://img.shields.io/badge/status-active%20dev-orange)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#)

> ⚠️ This project is under active development and may change significantly.

An agent-based model (ABM) of India’s **Carbon Credit Trading System (CCTS)**. It simulates heterogeneous firm behavior and a central carbon market to explore compliance rates, market stability, and economic impacts.

---

## ✨ Key Features

- **Agent-Based Modeling**: Firm agents optimize production, abatement, and trading.
- **Dynamic Market**: Price discovery from supply–demand, with **Market Stability Reserve (MSR)** interventions.
- **Regulatory Compliance**: Annual checks, credit expiration, and non-compliance penalties.
- **Adaptive Learning**: Agents adjust risk aversion and market confidence based on past performance.
- **Rich Data Collection**: MESA `DataCollector` at model and agent levels.
- **Analysis & Visualization**: Post-run analysis scripts produce plots and a summary report.

---

## 🗂️ Project Structure

```
CCTS-Model/
├─ ccts/                     # Main Python package
│  ├─ __init__.py
│  ├─ agent.py               # FirmAgent (behavior, trading, abatement)
│  ├─ market.py              # IndianCarbonMarket (price, MSR, compliance)
│  ├─ model.py               # CCTSModel (scheduler, orchestration, data)
│  ├─ analysis.py            # Post-simulation analysis & plotting
│  ├─ utils.py               # Helpers, dataclasses, typing
│  └─ config.py              # Config loader & defaults
├─ ccts_input/          # Required input CSVs
│  ├─ firms.csv
│  ├─ macc.csv
│  ├─ config.csv
│  └─ market_params.csv
├─ output/                   # Auto-created outputs (CSVs, PNGs, report)
├─ scripts/
│  └─ run_simulation.py      # CLI entry point
├─ requirements.txt
├─ ccts_simulation.log       # Generated at runtime
└─ README.md
```

---

## 🚀 Quickstart

```bash
# 1) Clone
git clone https://github.com/ankitsharmaMCET/CCTS-Model.git
cd CCTS-Model

# 2) Create & activate a Python 3.10+ environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Prepare input data
# Ensure firms.csv, macc.csv, config.csv, market_params.csv exist in ccts_input/

# 5) Run
python scripts/run_simulation.py ccts_input output
```

On completion, look in `output/` for:
- `model_timeseries.csv`, `agent_timeseries.csv` (example names; adjust to your actual outputs)
- Plots (`.png`)
- `summary_report.txt`
- Updated `ccts_simulation.log`

---

## ⚙️ Configuration

### `config.csv` (example)
| key                | value  | notes                                   |
|--------------------|--------|-----------------------------------------|
| steps              | 365    | Simulation steps (e.g., days)           |
| initial_carbon_price | 1200 | INR/tCO₂e (example)                     |
| seed               | 42     | RNG seed for reproducibility            |
| reporting_freq     | 7      | Collect/flush metrics every N steps     |

### `market_params.csv` (example)
| param                 | value | notes                                  |
|-----------------------|-------|----------------------------------------|
| msr_lower_threshold   | 0.8   | Trigger ratio for MSR injection        |
| msr_upper_threshold   | 1.2   | Trigger ratio for MSR withdrawal       |
| penalty_per_ton       | 2000  | INR/tCO₂e for non-compliance           |
| expiry_horizon_steps  | 365   | Credit validity horizon                |

---

## 🧩 Input Data Schemas (minimal guide)

### `firms.csv`
| firm_id | sector       | baseline_emissions | abatement_budget | risk_aversion | target_intensity | ... |
|---------|--------------|--------------------|------------------|---------------|------------------|-----|

### `macc.csv`
| firm_id | project_id | marginal_cost | max_potential_tons | lifetime_steps | ... |
|---------|------------|---------------|--------------------|----------------|-----|

> Tip: Validate IDs (`firm_id`) align across files.

---

## 🏃 Running Options

```bash
# Choose different inputs and output folder
python scripts/run_simulation.py ./ccts_input ./output/run_001

# With a specific RNG seed (if supported by config.csv)
python scripts/run_simulation.py ./ccts_input ./output/run_seed42
```

**Logging:** All major events are written to `ccts_simulation.log`. Increase verbosity via your logging config (inside `config.py` or environment variable, as implemented).

---

## 📈 Analysis & Visualization

After a run, use analysis helpers in `ccts/analysis.py` to:
- Plot carbon price evolution, total emissions vs cap
- Agent profits, abatement, and compliance stats
- Market depth and MSR interventions

Common outputs go to `output/plots/` and `output/summary_report.txt`.

---

## ✅ Reproducibility

- Pin dependencies via `requirements.txt`.
- Set `seed` in `config.csv`.
- Archive the exact input directory (`ccts_input/`) with your results.

---

## 🔧 Troubleshooting

- **Module not found / version conflicts**  
  Use a clean virtual environment and reinstall:
  ```bash
  rm -rf .venv
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Run fails due to CSV schema**  
  Check column names/dtypes against the schemas above and ensure no missing mandatory fields.

- **Empty or NaN in outputs**  
  Verify `steps > 0`, reasonable `initial_carbon_price`, and non-zero `baseline_emissions`.

- **Plots not generated**  
  Ensure matplotlib is installed and the script has permission to write into `output/`.

---

## 🧪 Tests (suggested)

Add unit tests under `tests/`:
- `test_market.py`: price updates, MSR boundaries, penalties
- `test_agent.py`: abatement choice, trading logic, learning updates
- `test_model.py`: scheduler order, data collection

Run with:
```bash
pytest -q
```

---

## 🤝 Contributing

Contributions are welcome!
1. Fork → feature branch → commit with tests → PR
2. Follow type hints / docstrings and run a formatter (`black`, `ruff`)
3. Add/adjust docs where behavior changes

---

## 📄 License

MIT — see `LICENSE` for details.

---

## 📌 Roadmap (high-level)

- Calibrated scenarios with real CCTS parameters
- Market microstructure options (e.g., auctions vs continuous trading)
- Endogenous baseline and sectoral coupling
- Sensitivity analysis harness
- Interactive dashboard (e.g., Mesa server / Streamlit)
