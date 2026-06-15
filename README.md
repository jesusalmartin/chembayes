# ChemBayes

**ChemBayes** is a Python library designed to streamline **experimental design (DoE)** and **Bayesian optimization** for scientific and chemical research. It leverages Gaussian Processes to model complex response surfaces and efficiently identify optimal conditions.

## 🚀 Key Features

* **Intelligent Sampling (QMC):** Generate initial experimental designs using Halton sequences (Quasi-Monte Carlo) for optimal space-filling coverage.
* **Automatic Preprocessing:** Integrated handling of numeric variables (scaling) and categorical variables (one-hot encoding).
* **Hyperparameter Tuning:** Automated Gaussian Process parameter adjustment (Matern + WhiteKernel) using Leave-One-Out (LOO) cross-validation and NLPD loss.
* **Bayesian Optimization:** Global maximum search using the Expected Improvement (EI) acquisition function.
* **Visual Diagnostics:** Automated generation of feature importance, partial dependence plots (1D and 2D), and model validation charts.

## 📦 Installation

You can install the latest stable version directly from PyPI:

```bash
pip install chembayes
```

Alternatively, for local development, you can install from the project root:

```bash
pip install .
```

Or for development mode:

```bash
pip install -e .
```

## 🛠️ Quick Start

### 1. Generating an Experimental Design (Sampling)

Define your search space with numeric and categorical parameters:

```python
import chembayes as cb

parameters = {
    'temperature': {'type': 'float', 'l_bound': 20.0, 'u_bound': 100.0},
    'time': {'type': 'int', 'l_bound': 5, 'u_bound': 60},
    'catalyst': {'type': 'categoric', 'categories': ['Pd', 'Ni', 'Cu']}
}

# Generate 20 optimal experimental points
df_experiments = cb.qmc_sampling(parameters, n_points=20)

# Visualize the distribution
cb.plot_qmc_sampling(df_experiments)
```

### 2. Bayesian Optimization

Once you have experimental data, find the optimal conditions:

```python
# Define input features and the target column
inputs = ['temperature', 'time', 'catalyst']
output = 'yield'

# Run the complete optimization pipeline
results = cb.optimize_experiment(
    df=df_data, 
    inputs=inputs, 
    output=output,
    n_tuning_trials=50,
    n_opt_trials=100
)

# Access the best found parameters
print(f"Best configuration: {results['best_params']}")
```

## 📂 Project Structure

* `src/chembayes/sampler.py`: Tools for QMC sample generation.
* `src/chembayes/optimizer.py`: Bayesian optimization engine and Gaussian Processes.
* `src/chembayes/__init__.py`: Public API exposure.
* `pyproject.toml`: Modern package configuration and dependencies.

## ✉️ Contact & Contribution

**Author:** Jesus Alberto Martin del Campo  
**Email:** j.a.martin-campo@hotmail.com  
**GitHub:** [jesusalmartin/chembayes](https://github.com/jesusalmartin/chembayes)
**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20701727.svg)](https://doi.org/10.5281/zenodo.20701727)
