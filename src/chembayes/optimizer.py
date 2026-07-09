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

class Optimizer:
    """
    Bayesian optimisation pipeline using Gaussian Processes.

    This class performs the complete optimisation workflow:
        1. Preprocess numeric features (StandardScaler) and categorical features (OneHotEncoder).
        2. Tune GP hyperparameters (Matern + WhiteKernel) via Leave‑One‑Out cross‑validation
           minimising the average negative log predictive density (NLPD). Tuning uses Optuna.
        3. Fit the final GP model with the best hyperparameters.
        4. Provide diagnostic plots: true vs predicted (with uncertainty), permutation importance.
        5. Perform Bayesian optimisation (Expected Improvement) to find input parameters that
           maximise the objective, again using Optuna.
        6. Reconstruct the best numeric parameters on the original scale.
        7. Generate partial dependence plots:
            - For numeric features: 1D line plots on the diagonal, 2D contour plots below.
            - For categorical features: bar charts showing average partial dependence per category.
        8. Print the best parameters and predicted objective.

    Attributes
    ----------
    data : pd.DataFrame
        The input dataset.
    inputs : list of str
        Column names used as input features.
    output : str or dict
        Target column name or weighted objective dictionary.
    n_tuning_trials : int
        Number of Optuna trials for hyperparameter tuning.
    n_opt_trials : int
        Number of Optuna trials for Bayesian optimisation.
    x : pd.DataFrame
        Input features after dropping rows with missing target/inputs.
    y : pd.Series
        Target values after dropping missing rows.
    _numeric_features : list of str
        Numeric input column names.
    _categoric_features : list of str
        Categorical input column names.
    _preprocessor : ColumnTransformer
        Sklearn transformer for scaling numeric and one‑hot encoding categorical columns.
    _model_features : ndarray of str
        Feature names after preprocessing.
    _numeric_model_features : list of str
        Preprocessed numeric feature names.
    _categoric_model_features : list of str
        Preprocessed categorical (one‑hot) feature names.
    _x_model : pd.DataFrame
        Preprocessed input features.
    _scaler : StandardScaler
        The fitted scaler for numeric features (for inverse transformation).
    model : GaussianProcessRegressor
        The final GP model fitted with optimal hyperparameters.
    opt_params : dict
        Best input parameters on the original scale.
    best_output : float
        Predicted objective at the optimal input.
    best_output_sigma : float
        Predictive standard deviation at the optimal input.
    model_score : float
        R² score of the final GP model on the training data.
    """
    def __init__(self, data, inputs, output, n_tuning_trials=100, n_opt_trials=100):
        """
        Initialises the Optimizer and runs the entire pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Input data. It must already be cleaned (e.g., NaNs handled) if needed.
        inputs : list of str
            Column names to use as input features.
        output : str or dict
            If str: name of the column to maximise.
            If dict: weights for a weighted objective (keys are column names,
                     values are weights). The weighted objective is computed as
                     the dot product of standardised columns and weights.
        n_tuning_trials : int, default=100
            Number of Optuna trials for GP hyperparameter tuning.
        n_opt_trials : int, default=100
            Number of Optuna trials for Bayesian optimisation (Expected Improvement).
        """
                
        self.data = data
        self.inputs = inputs
        self.output = output
        self.n_tuning_trials = n_tuning_trials
        self.n_opt_trials = n_opt_trials

        def _create_objective():
            """
            Internal helper to create a weighted objective column.

            Standardises the selected columns and computes a weighted sum.
            The result is stored in the 'objective' column of self.data.

            Returns
            -------
            pd.Series
                The weighted objective values.
            """
            cols = list(self.output.keys())
            subset = self.data[cols]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(subset)
            weight_series = pd.Series(self.output)
            objective = scaled_data @ weight_series.values
            return objective
        
        # Create reduced DataFrame with weighted objective if needed
        if isinstance(output, dict):
            self.data['objective'] = _create_objective()
            self.output = 'objective'


        # Drop rows with missing values in selected inputs or output
        model_df = self.data.dropna(subset=self.inputs+[self.output])

        self.x = model_df[self.inputs]
        self.y = model_df[self.output]

        # Identify numeric and categorical columns
        self._numeric_features = [col for col in self.x.columns if col in self.x.select_dtypes(include='number').columns]
        self._categoric_features = [col for col in self.x.columns if col in self.x.select_dtypes(include=['object', 'category', 'string']).columns]

        # Preprocess input columns using ColumnTransformer
        self._preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self._numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self._categoric_features)
            ])

        x_processed = self._preprocessor.fit_transform(self.x)
        self._model_features = self._preprocessor.get_feature_names_out()
        self._numeric_model_features = [f for f in self._model_features if f.startswith('num__')]
        self._categoric_model_features = [f for f in self._model_features if f.startswith('cat__')]
        self._x_model = pd.DataFrame(x_processed, columns=self._model_features)

        # Keep a reference to the scaler for inverse transformation later
        self._scaler = self._preprocessor.named_transformers_['num']

        # ----- Gaussian Process hyperparameter tuning (LOO-CV with NLPD) -----
        def _gp_tuning_objective(trial):
            """
            Optuna objective function for GP hyperparameter tuning.

            Suggests length_scale, noise_level, and Matern nu. Evaluates the
            average negative log predictive density (NLPD) via Leave‑One‑Out
            cross‑validation.

            Parameters
            ----------
            trial : optuna.Trial
                The current trial object.

            Returns
            -------
            float
                The mean NLPD over all LOO folds (to be minimised).
            """
            # Suggest hyperparameters
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
            noise_level = trial.suggest_float('noise_level', 1e-3, 10.0, log=True)

            nu_values = [0.5, 1.5, 2.5, np.inf]
            nu_idx = trial.suggest_int('nu_idx', 0, 3)
            nu = nu_values[nu_idx]

            # Leave-one-out cross-validation
            loo = LeaveOneOut()
            nlpds = []  # store negative log predictive densities

            for train_idx, test_idx in loo.split(self._x_model):
                x_train_df = self._x_model.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                x_test_df = self._x_model.iloc[test_idx]
                y_test = self.y.iloc[test_idx]

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
                gp.fit(x_train_df, y_train)

                # Predict on test point (mean and std)
                mu, sigma = gp.predict(x_test_df, return_std=True)
                mu = mu[0]
                sigma = sigma[0]

                # Avoid sigma=0 by clipping
                sigma = max(sigma, 1e-6)
                nlpd = 0.5 * np.log(2 * np.pi * sigma**2) + (y_test - mu)**2 / (2 * sigma**2)
                nlpds.append(nlpd)

            # Average NLPD across folds (to be minimized)
            return np.mean(nlpds)

        gp_tuning_study = optuna.create_study(direction='minimize')
        gp_tuning_study.optimize(_gp_tuning_objective, n_trials=n_tuning_trials)
        gp_best_params = gp_tuning_study.best_params

        # Build final GP with best hyperparameters
        nu_values = [0.5, 1.5, 2.5, np.inf]
        length_scale = gp_best_params['length_scale']
        noise_level = gp_best_params['noise_level']
        nu = nu_values[gp_best_params['nu_idx']]

        kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            optimizer=None,
            normalize_y=True,
            random_state=42
        )
        self.model.fit(self._x_model, self.y)

        # ----- Bayesian optimization (Expected Improvement) -----
        y_best = self.y.max()

        def _opt_objective(trial):
            """
            Optuna objective for Bayesian optimisation (Expected Improvement).

            Suggests values for all preprocessed numeric and categorical features,
            predicts the GP mean and standard deviation, and computes the Expected
            Improvement over the current best observed objective.

            Parameters
            ----------
            trial : optuna.Trial
                The current trial object.

            Returns
            -------
            float
                The Expected Improvement (to be maximised).
            """
            trial_dict = {}
            for feature in self._numeric_model_features:
                trial_dict[feature] = [trial.suggest_float(feature, self._x_model[feature].min(), self._x_model[feature].max())]
            for feature in self._categoric_features:
                ohe_group = [ohe_feature for ohe_feature in self._categoric_model_features if f'cat__{feature}' in ohe_feature]
                trial_category = trial.suggest_categorical(feature, self.x[feature].unique())
                for ohe_feature in ohe_group:
                    trial_dict[ohe_feature] = 1 if ohe_feature == f'cat__{feature}_{trial_category}' else 0
            trial_df = pd.DataFrame(trial_dict)
            trial_df = trial_df[self._model_features]

            # Predict mean and standard deviation
            mu, sigma = self.model.predict(trial_df, return_std=True)
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
        study.optimize(_opt_objective, n_trials=n_opt_trials)
        best_params = study.best_params

        # ----- Reconstruct best numeric values on original scale -----
        best_num_values_scaled = np.array([value for feature, value in best_params.items() if 'num__' in feature]).reshape(1, -1)
        best_num_values = self._scaler.inverse_transform(best_num_values_scaled)
        self.opt_params = {}
        for index, feature in enumerate(self._numeric_features):
            self.opt_params[feature] = best_num_values[0, index]
        for feature in self._categoric_features:
            self.opt_params[feature] = best_params[feature]

        # Predict objective at the optimal input
        opt_params_processed = self._preprocessor.transform(pd.DataFrame([self.opt_params]))
        opt_params_model = pd.DataFrame(opt_params_processed, columns=self._model_features)
        best_output, best_output_sigma = self.model.predict(opt_params_model, return_std=True)
        self.best_output = best_output[0]
        self.best_output_sigma = best_output_sigma[0]
        self.model_score = self.model.score(self._x_model, self.y)

    def summary(self):
        """
        Prints a summary of the optimisation results.

        Displays:
            - Model R² score.
            - Best input parameters (on the original scale).
            - Best predicted objective value and its uncertainty (±1σ).

        Returns
        -------
        None
        """
        print("\n--- Optimization results ---")
        print(f"Model score (R²): {self.model_score:.4f}")
        print("Best input parameters:")
        for feature, value in self.opt_params.items():
            print(f"  {feature}: {value}")
        print(f"Best predicted {self.output}: {self.best_output:.4f} ± {self.best_output_sigma:.4f}")

    def get_true_vs_pred_plot(self):
        """
        Plots true target values vs predicted values with 2σ uncertainty bands.

        The plot includes error bars representing twice the predictive standard
        deviation. A dashed diagonal line indicates perfect prediction.

        Returns
        -------
        matplotlib.figure.Figure: The generated figure object.
        """
        # True vs predicted with uncertainty
        y_pred, sigma = self.model.predict(self._x_model, return_std=True)
        y_upper = y_pred + 2*sigma
        y_lower = y_pred - 2*sigma
        all_values = np.concatenate([self.y, y_pred, y_upper, y_lower])

        fig, ax = plt.subplots()
        ax.errorbar(self.y, y_pred, yerr=2*sigma,fmt='o',
                    ecolor='lightgray', elinewidth=1, capsize=3,
                    #mfc='royalblue',
                    mec='white', alpha=0.8)
        ax.set_xlabel(f"True {self.output}")
        ax.set_ylabel(f"Predicted {self.output}")
        ax.set_title(f"{self.output}: true vs. predicted")
        ax.plot([all_values.min(), all_values.max()], [all_values.min(), all_values.max()], 'k--', linewidth=1)
        fig.tight_layout()
        return fig
    
    def true_vs_pred_plot(self):
        """
        Display the true target values vs predicted values with 2σ uncertainty
        bands plot.

        This is a convenience method that calls get_true_vs_pred_plot and shows
        the figure.

        Returns
        -------
        None
        """
        fig = self.get_true_vs_pred_plot()
        plt.show()

    def get_permutation_importance_plot(self):
        """
        Plots permutation importance of the preprocessed features.

        Permutation importance is computed using R² as the scoring metric over
        30 repetitions. The top 15 features are displayed in a horizontal bar plot.

        Returns
        -------
        matplotlib.figure.Figure: The generated figure object.
        """
        # Permutation importance
        permutation_result = permutation_importance(self.model, self._x_model, self.y, scoring='r2', n_repeats=30, random_state=42)
        importances = permutation_result.importances_mean
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = self._model_features[sorted_idx]
        fig, ax = plt.subplots()
        ax.barh(sorted_features[:15], sorted_importances[:15])
        ax.invert_yaxis()
        ax.set_xlabel("Permutation importance")
        fig.tight_layout()
        return fig
    
    def permutation_importance_plot(self):
        """
        Display the permutation importance of the preprocessed features plot.

        This is a convenience method that calls get_permutation_importance_plot and shows
        the figure.

        Returns
        -------
        None
        """
        fig = self.get_permutation_importance_plot()
        plt.show()

    def get_partial_dependence_plot(self):
        """
        Generates partial dependence plots for both numeric and categorical features.

        For numeric features:
            - Diagonal subplots show 1D partial dependence with the optimal value
              marked as a vertical line.
            - Below‑diagonal subplots show 2D contour plots of the partial dependence
              with the optimal point marked.
            - Above‑diagonal subplots are hidden.

        For categorical features:
            - Separate bar charts show the average partial dependence for each category
              of each categorical feature.

        Returns
        -------
        list of matplotlib.figure.Figure
            A list containing the generated figure objects (e.g., one figure for 
            numeric features matrix, and/or individual figures for categorical features).
        """
        figs = []
        # Partial dependence for numeric features: 1D on diagonal, 2D below diagonal
        if self._numeric_features:
            n_plots = len(self._numeric_model_features)
            grid_point_number = 25

            fig, axes = plt.subplots(n_plots, n_plots, figsize=(3*n_plots, 3*n_plots))

            for i, feature_i in enumerate(self._numeric_model_features):
                for j, feature_j in enumerate(self._numeric_model_features):
                    ax = axes[i, j] if n_plots > 1 else axes

                    if i == j:
                        # 1D partial dependence on diagonal
                        part_dep = partial_dependence(
                            self.model,
                            X=self._x_model,
                            features=[feature_i],
                            custom_values={feature_i: np.linspace(self._x_model[feature_i].min(), self._x_model[feature_i].max(), grid_point_number)}
                        )
                        xi = part_dep['grid_values'][0]
                        yi = part_dep['average'][0]

                        # Transform back to original scale for labeling
                        original_feature_i = feature_i[5:]
                        original_idx_i = self._numeric_features.index(original_feature_i)
                        dummy_array = np.zeros((grid_point_number, len(self._numeric_features)))
                        dummy_array[:, original_idx_i] = xi
                        transformed_array = self._scaler.inverse_transform(dummy_array)
                        xi = transformed_array[:, original_idx_i]

                        ax.plot(xi, yi)
                        ax.axvline(x=self.opt_params[original_feature_i], color='black', alpha=0.5)
                        ax.set_xlabel(original_feature_i)
                        ax.set_ylabel(self.output)

                    elif j < i:
                        # 2D partial dependence below diagonal
                        part_dep = partial_dependence(
                            self.model,
                            X=self._x_model,
                            features=[feature_i, feature_j],
                            custom_values={
                                feature_i: np.linspace(self._x_model[feature_i].min(), self._x_model[feature_i].max(), grid_point_number),
                                feature_j: np.linspace(self._x_model[feature_j].min(), self._x_model[feature_j].max(), grid_point_number)
                            }
                        )

                        xi = part_dep['grid_values'][0]
                        yi = part_dep['grid_values'][1]
                        zi = part_dep['average'][0].T

                        original_feature_i = feature_i[5:]
                        original_feature_j = feature_j[5:]

                        original_idx_i = self._numeric_features.index(original_feature_i)
                        original_idx_j = self._numeric_features.index(original_feature_j)

                        dummy_array = np.zeros((grid_point_number, len(self._numeric_features)))
                        dummy_array[:, original_idx_i] = xi
                        dummy_array[:, original_idx_j] = yi

                        transformed_array = self._scaler.inverse_transform(dummy_array)
                        xi = transformed_array[:, original_idx_i]
                        yi = transformed_array[:, original_idx_j]

                        ax.pcolormesh(xi, yi, zi, cmap='coolwarm')
                        ax.scatter(self.opt_params[original_feature_i], self.opt_params[original_feature_j], color='black', alpha=0.5)
                        ax.set_xlabel(original_feature_i)
                        ax.set_ylabel(original_feature_j)
                    else:
                        # Upper triangular part: hide axes
                        ax.axis('off')
            fig.tight_layout()
            figs.append(fig)

        # Partial dependence for categorical features: bar charts per category
        if self._categoric_features:

            n_plots = len(self._categoric_features)
            fig, axes = plt.subplots(n_plots, 1, figsize=(6, 5*n_plots))

            for i, feature in enumerate(self._categoric_features):
                ax = axes[i] if n_plots > 1 else axes

                # Get all one-hot columns for this original feature
                ohe_features = [f for f in self._categoric_model_features if f.startswith(f'cat__{feature}')]

                x_labels = []
                y_values = []

                for ohe_feature in ohe_features:
                    part_dep = partial_dependence(self.model, X=self._x_model, features=[ohe_feature])
                    yi = part_dep['average'][0]
                    category = ohe_feature[len(f'cat__{feature}_'):]
                    y_values.append(yi[-1])  # partial dependence value for this category
                    x_labels.append(category)

                ax.bar(x_labels, y_values)
                ax.set_xlabel(feature)
                ax.set_ylabel(self.output)
                ax.tick_params(axis='x', labelrotation=45)
            fig.tight_layout()
            figs.append(fig)
        return figs
    
    def partial_dependence_plot(self):
        """
        Display partial dependence plots for both numeric and categorical features.

        This is a convenience method that calls get_partial_dependence_plot and shows
        the figures.

        Returns
        -------
        None
        """
        figs = self.get_partial_dependence_plot()
        for fig in figs:
            plt.show()