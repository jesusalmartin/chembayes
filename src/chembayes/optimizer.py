import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.inspection import permutation_importance, partial_dependence
from scipy.stats import norm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def create_objective(df, weights):
    """
    Create a weighted objective column.
    """
    cols = list(weights.keys())
    subset = df[cols]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(subset)
    weight_series = pd.Series(weights)
    objective = scaled_data @ weight_series.values
    return objective

def optimize_experiment(df, inputs, output, n_tuning_trials=100, n_opt_trials=100, plot=True):
    """
    Run the complete optimization pipeline using Gaussian Processes and Bayesian optimization.

    The pipeline consists of:
        1. Preprocessing numeric features (StandardScaler) and categorical features (OneHotEncoder).
        2. Tuning GP hyperparameters (Matern + WhiteKernel) via Leave-One-Out cross-validation
           minimizing the average negative log predictive density (NLPD). Tuning uses Optuna.
        3. Fitting the final GP model with the best hyperparameters.
        4. (Optional) Diagnostic plots: true vs predicted (colored by uncertainty), permutation importance.
        5. Bayesian optimization (Expected Improvement) to find the input parameters that maximize
           the objective. This also uses Optuna.
        6. Reconstructing the best numeric parameters on the original scale.
        7. (Optional) Partial dependence plots:
            - For numeric features: 1D line plots on the diagonal, 2D contour plots below the diagonal.
            - For categorical features: bar charts showing the average partial dependence per category.
        8. Printing the best parameters and predicted objective.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. It must already be cleaned (e.g., NaNs handled) if needed.
    inputs : list of str
        Column names to use as input features.
    output : str or dict
        If str: name of the column to maximize.
        If dict: weights for weighted objective (passed to create_objective).
    n_tuning_trials : int, default=100
        Number of Optuna trials for GP hyperparameter tuning.
    n_opt_trials : int, default=100
        Number of Optuna trials for Bayesian optimization (Expected Improvement).
    plot : bool, default=True
        Whether to show diagnostic plots:
            - true vs predicted (with uncertainty)
            - permutation importance
            - partial dependence (1D/2D for numeric, bar charts for categorical)

    Returns
    -------
    dict
        A dictionary containing:
        - 'best_params': dict of optimal input parameters (on original scale).
        - 'best_value': float, predicted objective at best_params.
        - 'model': fitted GaussianProcessRegressor.
        - 'preprocessor': fitted ColumnTransformer.
        - 'X': preprocessed input matrix (numpy array).
        - 'y': objective array.
    """
    # Create reduced DataFrame with weighted objective if needed
    if isinstance(output, dict):
        df['objective'] = create_objective(df, output)
        output = 'objective'

    # Drop rows with missing values in selected inputs or output
    model_df = df.dropna(subset=inputs+[output])

    X_df = model_df[inputs]
    y = model_df[output]

    # Identify numeric and categorical columns
    numeric_features = [col for col in X_df.columns if col in X_df.select_dtypes(include='number').columns]
    categorical_features = [col for col in X_df.columns if col in X_df.select_dtypes(include=['object', 'category', 'string']).columns]

    # Preprocess input columns using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X = preprocessor.fit_transform(X_df)
    model_features = preprocessor.get_feature_names_out()
    X_model = pd.DataFrame(X, columns=model_features)

    # Keep a reference to the scaler for inverse transformation later
    scaler = preprocessor.named_transformers_['num']

    # ----- Gaussian Process hyperparameter tuning (LOO-CV with NLPD) -----
    def gp_tuning_objective(trial):
        # Suggest hyperparameters
        length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
        noise_level = trial.suggest_float('noise_level', 1e-3, 10.0, log=True)

        nu_values = [0.5, 1.5, 2.5, np.inf]
        nu_idx = trial.suggest_int('nu_idx', 0, 3)
        nu = nu_values[nu_idx]

        # Leave-one-out cross-validation
        loo = LeaveOneOut()
        nlpds = []  # store negative log predictive densities

        for train_idx, test_idx in loo.split(X_df):
            X_train_df = X_model.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test_df = X_model.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Create kernel with current hyperparameters
            kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)

            # Build GP (no internal optimization, fixed kernel parameters)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,                # we use WhiteKernel instead
                optimizer=None,
                normalize_y=True,
                random_state=42
            )
            gp.fit(X_train_df, y_train)

            # Predict on test point (mean and std)
            mu, sigma = gp.predict(X_test_df, return_std=True)
            mu = mu[0]
            sigma = sigma[0]

            # Avoid sigma=0 by clipping
            sigma = max(sigma, 1e-6)
            nlpd = 0.5 * np.log(2 * np.pi * sigma**2) + (y_test - mu)**2 / (2 * sigma**2)
            nlpds.append(nlpd)

        # Average NLPD across folds (to be minimized)
        return np.mean(nlpds)

    gp_tuning_study = optuna.create_study(direction='minimize')
    gp_tuning_study.optimize(gp_tuning_objective, n_trials=n_tuning_trials)
    gp_best_params = gp_tuning_study.best_params

    # Build final GP with best hyperparameters
    nu_values = [0.5, 1.5, 2.5, np.inf]
    length_scale = gp_best_params['length_scale']
    noise_level = gp_best_params['noise_level']
    nu = nu_values[gp_best_params['nu_idx']]

    kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        optimizer=None,
        normalize_y=True,
        random_state=42
    )
    model.fit(X_model, y)


    # ----- Bayesian optimization (Expected Improvement) -----
    y_best = y.max()

    def objective(trial):
        numeric_model_features = [f for f in model_features if f.startswith('num__')]
        categorical_model_features = [f for f in model_features if f.startswith('cat__')]
        trial_dict = {}
        for feature in numeric_model_features:
            trial_dict[feature] = [trial.suggest_float(feature, X_model[feature].min(), X_model[feature].max())]
        for feature in categorical_features:
            ohe_group = [ohe_feature for ohe_feature in categorical_model_features if f'cat__{feature}' in ohe_feature]
            trial_category = trial.suggest_categorical(feature, X_df[feature].unique())
            for ohe_feature in ohe_group:
                trial_dict[ohe_feature] = 1 if ohe_feature == f'cat__{feature}_{trial_category}' else 0
        trial_df = pd.DataFrame(trial_dict)
        trial_df = trial_df[model_features]

        # Predict mean and standard deviation
        mu, sigma = model.predict(trial_df, return_std=True)
        mu = mu[0]
        sigma = sigma[0]

        # Expected Improvement for maximization
        if sigma <= 0:
            return 0.0
        imp = mu - y_best
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_opt_trials)
    best_params = study.best_params

    # ----- Reconstruct best numeric values on original scale -----
    best_num_values_scaled = np.array([value for feature, value in best_params.items() if 'num__' in feature]).reshape(1, -1)
    best_num_values = scaler.inverse_transform(best_num_values_scaled)
    opt_params = {}
    for index, feature in enumerate(numeric_features):
        opt_params[feature] = best_num_values[0, index]
    for feature in categorical_features:
        opt_params[feature] = best_params[feature]

    # ----- Diagnostic plots (if requested) -----
    if plot:
        # True vs predicted with uncertainty
        y_pred, sigma = model.predict(pd.DataFrame(X, columns=model.feature_names_in_), return_std=True)
        plt.figure()
        sc = plt.scatter(y, y_pred, c=sigma, alpha=0.6, cmap='coolwarm', edgecolor='k', linewidth=0.5)
        plt.colorbar(sc, label=f'Prediction std ({output})')
        plt.xlabel(f"True {output}")
        plt.ylabel(f"Predicted {output}")
        plt.title(f"{output}: true vs. predicted (colored by uncertainty)")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=1)
        plt.tight_layout()
        plt.show()

        # Permutation importance
        permutation_result = permutation_importance(model, X_model, y, scoring='r2', n_repeats=30, random_state=42)
        importances = permutation_result.importances_mean
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = model.feature_names_in_[sorted_idx]
        plt.figure()
        plt.barh(sorted_features[:15], sorted_importances[:15])
        plt.gca().invert_yaxis()
        plt.xlabel("Permutation importance")
        plt.tight_layout()
        plt.show()

        # Partial dependence for numeric features: 1D on diagonal, 2D below diagonal
        if numeric_features:
            numeric_model_features = [f for f in model_features if f.startswith('num__')]

            n_plots = len(numeric_model_features)
            grid_point_number = 25

            fig, axes = plt.subplots(n_plots, n_plots, figsize=(3*n_plots, 3*n_plots))

            for i, feature_i in enumerate(numeric_model_features):
                for j, feature_j in enumerate(numeric_model_features):
                    ax = axes[i, j] if n_plots > 1 else axes

                    if i == j:
                        # 1D partial dependence on diagonal
                        part_dep = partial_dependence(
                            model,
                            X=X_model,
                            features=[feature_i],
                            custom_values={feature_i: np.linspace(X_model[feature_i].min(), X_model[feature_i].max(), grid_point_number)}
                        )
                        xi = part_dep['grid_values'][0]
                        yi = part_dep['average'][0]

                        # Transform back to original scale for labeling
                        original_feature_i = feature_i[5:]
                        original_idx_i = numeric_features.index(original_feature_i)
                        dummy_array = np.zeros((grid_point_number, len(numeric_features)))
                        dummy_array[:, original_idx_i] = xi
                        transformed_array = scaler.inverse_transform(dummy_array)
                        xi = transformed_array[:, original_idx_i]

                        ax.plot(xi, yi)
                        ax.axvline(x=opt_params[original_feature_i], color='black', alpha=0.5)
                        ax.set_xlabel(original_feature_i)
                        ax.set_ylabel(output)

                    elif j < i:
                        # 2D partial dependence below diagonal
                        part_dep = partial_dependence(
                            model,
                            X=X_model,
                            features=[feature_i, feature_j],
                            custom_values={
                                feature_i: np.linspace(X_model[feature_i].min(), X_model[feature_i].max(), grid_point_number),
                                feature_j: np.linspace(X_model[feature_j].min(), X_model[feature_j].max(), grid_point_number)
                            }
                        )

                        xi = part_dep['grid_values'][0]
                        yi = part_dep['grid_values'][1]
                        zi = part_dep['average'][0].T

                        original_feature_i = feature_i[5:]
                        original_feature_j = feature_j[5:]

                        original_idx_i = numeric_features.index(original_feature_i)
                        original_idx_j = numeric_features.index(original_feature_j)

                        dummy_array = np.zeros((grid_point_number, len(numeric_features)))
                        dummy_array[:, original_idx_i] = xi
                        dummy_array[:, original_idx_j] = yi

                        transformed_array = scaler.inverse_transform(dummy_array)
                        xi = transformed_array[:, original_idx_i]
                        yi = transformed_array[:, original_idx_j]

                        ax.pcolormesh(xi, yi, zi, cmap='coolwarm')
                        ax.scatter(opt_params[original_feature_i], opt_params[original_feature_j], color='black', alpha=0.5)
                        ax.set_xlabel(original_feature_i)
                        ax.set_ylabel(original_feature_j)
                    else:
                        # Upper triangular part: hide axes
                        ax.axis('off')
            fig.tight_layout()
            plt.show()

        # Partial dependence for categorical features: bar charts per category
        if categorical_features:
            categorical_model_features = [f for f in model_features if f.startswith('cat__')]

            n_plots = len(categorical_features)
            fig, axes = plt.subplots(n_plots, 1, figsize=(6, 5*n_plots))

            for i, feature in enumerate(categorical_features):
                ax = axes[i] if n_plots > 1 else axes

                # Get all one-hot columns for this original feature
                ohe_features = [f for f in categorical_model_features if f.startswith(f'cat__{feature}')]

                x_labels = []
                y_values = []

                for ohe_feature in ohe_features:
                    part_dep = partial_dependence(model, X=X_model, features=[ohe_feature])
                    yi = part_dep['average'][0]
                    category = ohe_feature[len(f'cat__{feature}_'):]
                    y_values.append(yi[-1])  # partial dependence value for this category
                    x_labels.append(category)

                ax.bar(x_labels, y_values)
                ax.set_xlabel(feature)
                ax.set_ylabel(output)
                ax.tick_params(axis='x', labelrotation=45)
            fig.tight_layout()
            plt.show()

    # Predict objective at the optimal input
    opt_params_df = pd.DataFrame([opt_params])
    opt_params_input = preprocessor.transform(opt_params_df)
    best_output = model.predict(pd.DataFrame(opt_params_input, columns=model.feature_names_in_))[0]

    print("\n--- Optimization results ---")
    print(f"Model score (R²): {model.score(pd.DataFrame(X, columns=model.feature_names_in_), y):.4f}")
    print("Best input parameters:")
    for feature, value in opt_params.items():
        print(f"  {feature}: {value}")
    print(f"Best predicted {output}: {best_output:.4f}")

    # ----- Return all important objects -----
    return {
        'best_params': opt_params,
        'best_value': best_output,
        'model': model,
        'preprocessor': preprocessor,
        'X': X,
        'y': y
    }