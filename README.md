# chembayes

A simple, no‑frills Bayesian optimization library for experimental design, based on Gaussian Processes.  
It wraps a **optimization workflow** into a single function that:

- Preprocesses numeric and categorical inputs (scaling + one‑hot encoding)
- Tunes GP hyperparameters using Leave‑One‑Out cross‑validation (NLPD loss)
- Provides diagnostic plots (true vs predicted, permutation importance, partial dependence)
- Performs Bayesian optimization using Expected Improvement to find the optimal inputs