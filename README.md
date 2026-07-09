# ChemBayes

**ChemBayes** is a Python library designed to streamline **experimental design (DoE)** and **Bayesian optimization** for scientific and chemical research. It leverages Gaussian Processes to model complex response surfaces and efficiently identify optimal conditions.

## 🚀 Key Features

* **Intelligent Sampling (QMC):** Generate initial experimental designs using Halton sequences (Quasi-Monte Carlo) for optimal space-filling coverage. Supports integer, float, and categorical parameters.
* **Automatic Preprocessing:** Integrated handling of numeric variables (scaling) and categorical variables (one-hot encoding).
* **Hyperparameter Tuning:** Automated Gaussian Process parameter adjustment (Matern + WhiteKernel) using Leave‑One‑Out (LOO) cross‑validation and NLPD loss.
* **Bayesian Optimization:** Global maximum search using the Expected Improvement (EI) acquisition function. Supports both single‑objective and weighted multi‑objective optimization.
* **Visual Diagnostics:** Automated generation of model validation charts (true vs predicted with uncertainty), feature importance (permutation importance), and partial dependence plots (1D, 2D, and categorical bar charts).

## 📦 Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install chembayes
```

### Option 2: Install from source (for development or latest version)

```bash
git clone https://github.com/jesusalmartin/chembayes.git
cd chembayes
pip install -e .
```

## 🛠️ Quick Start

### 1. Generating an Experimental Design (Sampling)

Define your search space with numeric and categorical parameters:

```python
from chembayes import Sampler

# Create a sampler instance
sampler = Sampler()

# Add parameters
sampler.add_float('temperature', 20.0, 100.0)
sampler.add_int('time', 5, 60)
sampler.add_categoric('catalyst', ['Pd', 'Ni', 'Cu'])

# Generate 20 optimal experimental points
sampler.qmc_sampling(n_points=20)

# Get the sample as a DataFrame
df_experiments = sampler.get_df()

# Visualize the distribution of numeric parameters
sampler.plot()
```

You can also define parameters using a dictionary:

```python
params = {
    'temperature': {'type': 'float', 'l_bound': 20.0, 'u_bound': 100.0},
    'time': {'type': 'int', 'l_bound': 5, 'u_bound': 60},
    'catalyst': {'type': 'categoric', 'categories': ['Pd', 'Ni', 'Cu']}
}

sampler.set_params(params)
sampler.qmc_sampling(20)
```

### 2. Bayesian Optimization

Once you have experimental data, find the optimal conditions:

#### Single Output Optimization

```python
from chembayes import Optimizer

# Define input features and the target column
inputs = ['temperature', 'time', 'catalyst']
output = 'yield'

# Run the complete optimization pipeline
opt = Optimizer(
    data=df_data,
    inputs=inputs,
    output=output,
    n_tuning_trials=50,
    n_opt_trials=100
)

# Print the best parameters and predicted objective
opt.summary()

# Visualize model performance and feature importance
opt.true_vs_pred_plot()
opt.permutation_importance_plot()

# Generate partial dependence plots to understand the response surface
opt.partial_dependence_plot()
```

#### Weighted Multi‑Output Optimization

If you have multiple response variables and want to optimize a weighted combination:

```python
# Define weights for each output (higher weight = more importance)
outputs = {
    'yield': 0.7,
    'selectivity': 0.2,
    'cost': 0.1
}

opt = Optimizer(
    data=df_data,
    inputs=inputs,
    output=outputs,
    n_tuning_trials=50,
    n_opt_trials=100
)

# Results automatically use the weighted objective
opt.summary()
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

## Citation