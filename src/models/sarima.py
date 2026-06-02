import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product


class SARIMAModel:
    """
    Classe responsável pelo build e otimização do modelo SARIMA.
    Pré-processamento, split, validação cruzada, bootstrap e visualizações
    são realizados externamente no notebook principal.
    """

    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)):
        """
        Parameters
        ----------
        order : tuple
            Parâmetros (p, d, q) do componente ARIMA.
        seasonal_order : tuple
            Parâmetros sazonais (P, D, Q, s).
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def build(self, y_train: np.ndarray):
        """
        Ajusta o modelo SARIMA aos dados de treino.

        Parameters
        ----------
        y_train : array-like
            Série temporal de casos para treino.

        Returns
        -------
        self
        """
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        Gera previsões para os próximos `steps` períodos.

        Parameters
        ----------
        steps : int
            Número de passos à frente a prever.

        Returns
        -------
        np.ndarray
            Previsões (valores não-negativos).
        """
        if self.model_fit is None:
            raise ValueError("Modelo não foi treinado. Chame build() primeiro.")
        y_pred = self.model_fit.forecast(steps=steps)
        return np.maximum(y_pred.values, 0)

    def optimize(
        self,
        y_train: np.ndarray,
        p_range=(0, 3),
        d_range=(0, 2),
        q_range=(0, 3),
        P_range=(0, 2),
        D_range=(0, 1),
        Q_range=(0, 2),
        s: int = 12,
        metric: str = "aic"
    ):
        """
        Busca em grade (grid search) pelos melhores hiperparâmetros SARIMA
        usando AIC ou BIC como critério.

        Parameters
        ----------
        y_train : array-like
            Série temporal de treino.
        p_range, d_range, q_range : tuple
            Intervalos (min, max_exclusive) para p, d, q.
        P_range, D_range, Q_range : tuple
            Intervalos para componentes sazonais.
        s : int
            Período sazonal (12 para dados mensais).
        metric : str
            Critério de seleção: 'aic' ou 'bic'.

        Returns
        -------
        dict
            Melhor configuração encontrada com chaves 'order',
            'seasonal_order' e o valor da métrica escolhida.
        """
        best_score = np.inf
        best_cfg = None

        p_vals = range(*p_range)
        d_vals = range(*d_range)
        q_vals = range(*q_range)
        P_vals = range(*P_range)
        D_vals = range(*D_range)
        Q_vals = range(*Q_range)

        configs = list(product(p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals))
        print(f"[SARIMAModel] Testando {len(configs)} configurações (métrica={metric.upper()})...")

        for (p, d, q, P, D, Q) in configs:
            try:
                m = SARIMAX(
                    y_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = m.fit(disp=False)
                score = fit.aic if metric == "aic" else fit.bic
                if score < best_score:
                    best_score = score
                    best_cfg = {
                        "order": (p, d, q),
                        "seasonal_order": (P, D, Q, s),
                        metric: score
                    }
            except Exception:
                continue

        if best_cfg:
            self.order = best_cfg["order"]
            self.seasonal_order = best_cfg["seasonal_order"]
            print(f"[SARIMAModel] Melhor config → order={self.order}, "
                  f"seasonal_order={self.seasonal_order}, {metric.upper()}={best_score:.2f}")

        return best_cfg