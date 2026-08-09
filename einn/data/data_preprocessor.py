import torch
from einn.ode.base_ode_model import BaseODEModel


class DataPreprocessor:
    """
    Class for scaling the data and generating auxillary trajectories for given model
    """
    def __init__(self):
        self.scalers = {}  # dictionary for parameters of scaling

    def scale_data(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """
        Scale the data blabla
        :param x:
        :param y:
        :param t:
        :return:
        """
        # TODO: ez inkább legyen sklearn-el csinálva?
        # Skálázás a célváltozón (Z-score)
        y_mean, y_std = y.mean(), y.std()
        y_scaled = (y - y_mean) / (y_std + 1e-8)
        self.scalers['y_mean'] = y_mean
        self.scalers['y_std'] = y_std

        # X skálázása (Z-score, dimenziófüggetlenül is megírható)
        x_mean, x_std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        x_scaled = (x - x_mean) / (x_std + 1e-8)
        self.scalers['X_mean'] = x_mean
        self.scalers['X_std'] = x_std

        # t skálázása [0, 1] intervallumra (Min-Max)
        t_min, t_max = t.min(), t.max()
        t_scaled = (t - t_min) / (t_max - t_min + 1e-8)
        self.scalers['t_min'] = t_min
        self.scalers['t_max'] = t_max

        return x_scaled, y_scaled, t_scaled

    def inverse_scale_predictions(self, s_future: torch.Tensor) -> torch.Tensor:
        """
        Rescales the predictions blabla
        :param s_future:
        :return:
        """
        if 'y_mean' not in self.scalers or 'y_std' not in self.scalers:
            raise ValueError("A predikciók inverz skálázása előtt a scale_data metódust le kell futtatni.")

        # Feltételezve, hogy az s_future első oszlopa / dimenziója felel meg az y-nak
        return (s_future * self.scalers['y_std']) + self.scalers['y_mean']

    def generate_aux_targets(self, t: torch.Tensor, ode_model: BaseODEModel) -> torch.Tensor:
        """
        Generates auxillary targets for given ode model
        :param t:
        :param ode_model:
        :return:
        """
        # TODO: Valós ODE solver implementálása (pl. torchdiffeq) fix paraméterekkel.
        print("Attention: generation of aux targets is currently in mock mode.")
        N = t.shape[1]
        # TODO: általánosítani "bármilyen modellre"
        return torch.zeros(1, N, 5)  # 5 az ODE állapotok száma (SEIRM)