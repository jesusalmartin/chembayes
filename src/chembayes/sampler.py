import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

class Sampler:
    """
    A class for generating parameter samples using Quasi-Monte Carlo (Halton sequence).

    The sampler supports integer, float, and categorical parameters. It provides
    methods to add parameters, generate a space-filling sample, retrieve the
    sample as a DataFrame, and visualise the sample distribution.

    Attributes
    ----------
    params : dict
        Dictionary defining the parameters and their bounds/types.
    sample : dict or None
        The generated sample after calling qmc_sampling().
    n_points : int or None
        Number of sample points.
    force_edges : bool or None
        Whether the sample was forced to cover the edges.
    qmc_sampler : qmc.Halton or None
        The Halton sampler instance.
    """

    def __init__(self, params: dict = {}):
        """
        Initialises the sampler with an optional parameter dictionary.

        Parameters
        ----------
        params : dict, optional
            Dictionary defining parameters (see set_params() for structure).
        """
        self.params = params
        
    def set_params(self, params: dict):
        """
        Replaces the current parameter dictionary with a new one.

        Parameters
        ----------
        params : dict
            Dictionary mapping parameter names to their definitions.
            Each definition should contain:
                - 'type': 'int', 'float', or 'categoric'
                - 'l_bound' and 'u_bound' for numeric types
                - 'categories' (list) for categorical type
        """
        self.params = params
    
    def add_int(self,name: str, l_bound: int, u_bound: int):
        """
        Adds an integer parameter to the sampler.

        Parameters
        ----------
        name : str
            Parameter name.
        l_bound : int
            Lower bound.
        u_bound : int
            Upper bound.
        """
        self.params[name] = {
            'type': 'int',
            'l_bound': l_bound,
            'u_bound': u_bound
        }
    
    def add_float(self, name: str, l_bound: float, u_bound: float):
        """
        Adds a float parameter to the sampler.

        Parameters
        ----------
        name : str
            Parameter name.
        l_bound : float
            Lower bound.
        u_bound : float
            Upper bound.
        """
        self.params[name] = {
            'type': 'float',
            'l_bound': l_bound,
            'u_bound': u_bound
        }
    
    def add_categoric(self, name: str, categories: list[str]):
        """
        Adds a categorical parameter to the sampler.

        Parameters
        ----------
        name : str
            Parameter name.
        categories : list
            List of possible category values (strings).
        """
        self.params[name] = {
            'type': 'categoric',
            'categories': categories
        }

    def qmc_sampling(self, n_points: int, force_edges: bool = True):
        """
        Generates a Quasi-Monte Carlo sample using a Halton sequence.

        The sample is scaled to the bounds of each parameter. If force_edges is True,
        the generated [0,1] sample is scaled so that its minimum and maximum map to
        the parameter bounds, ensuring that the sample covers the full range.
        If force_edges is False, the [0,1] sample is used directly (no edge forcing).

        The resulting sample is stored in the attribute `self.sample` as a dictionary
        of lists, one per parameter.

        Parameters
        ----------
        n_points : int
            Number of sample points to generate.
        force_edges : bool, default=True
            If True, rescale each parameter dimension so that the sample's min and max
            exactly match the parameter bounds. If False, use the raw [0,1] scaling.

        Returns
        -------
        None
            The sample is stored internally and can be retrieved via get_df().

        Raises
        ------
        TypeError
            If force_edges is not a boolean.
        ValueError
            If a parameter has an unsupported type.
        """
        self.n_points = n_points
        self.force_edges = force_edges

        self.qmc_sampler = qmc.Halton(d=len(self.params), seed=42)
        sample = self.qmc_sampler.random(n=n_points)

        sample_dict = {}
        for index, p in enumerate(self.params.keys()):

            values = sample[:, index]

            if force_edges:
                v_min, v_max = values.min(), values.max()
            elif force_edges==False:
                v_min, v_max = 0.0, 1.0
            else:
                return print('force_edges must be True or False')

            p_type = self.params[p]['type']

            if p_type == 'float':
                l_bound = self.params[p]['l_bound']
                u_bound = self.params[p]['u_bound']
                scaled = l_bound + (values - v_min) * (u_bound - l_bound) / (v_max - v_min)
                sample_dict[p] = scaled

            elif p_type == 'int':
                l_bound = self.params[p]['l_bound']
                u_bound = self.params[p]['u_bound']
                scaled = l_bound + (values - v_min) * (u_bound - l_bound) / (v_max - v_min)
                scaled = np.round(scaled).astype(int)
                sample_dict[p] = scaled

            elif p_type == 'categoric':
                v_min, v_max = values.min(), values.max()
                categories = self.params[p]['categories']
                l_bound = 0
                u_bound = len(categories)-1
                scaled = l_bound + (values - v_min) * (u_bound - l_bound) / (v_max - v_min)
                scaled = np.round(scaled).astype(int).tolist()
                scaled = [categories[v] for v in scaled]
                sample_dict[p] = scaled
            else:
                return print(f'{p}: type must be float, int or categoric')
        
        self.sample = sample_dict

    def get_df(self):
        """
        Returns the generated sample as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns corresponding to parameter names and rows
            representing sample points.
        """
        return pd.DataFrame(self.sample)
    
    def plot(self):
        """
        Creates a scatter plot matrix of the numeric parameters in the sample.

        This visualises the space-filling properties of the QMC sampling by
        plotting each numeric parameter against every other numeric parameter
        in a grid of subplots. Only parameters of type 'float' or 'int' are
        included; categorical parameters are ignored.

        The plot is displayed using matplotlib.pyplot.show().

        Returns
        -------
        None
        """
        num_params = [p for p in self.params.keys() if self.params[p]['type'] in ['float', 'int']]
        n_num_params = len(num_params)

        plt.figure(figsize=(3*n_num_params, 3*n_num_params))
        plot = 1
        for i, param_1 in enumerate(num_params):
            for j, param_2 in enumerate(num_params):
                if i >= j:
                    plt.subplot(n_num_params, n_num_params, plot)
                    plt.scatter(self.sample[param_1], self.sample[param_2])
                    plt.xlabel(param_1)
                    plt.ylabel(param_2)
                plot +=1
        plt.tight_layout()
        plt.show()