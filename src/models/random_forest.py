import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid


class RandomForestModel:
    """
    Classe responsável pelo build e otimização do modelo Random Forest
    para classificação binária de surtos.
    Pré-processamento, split, validação cruzada, bootstrap e visualizações
    são realizados externamente no notebook principal.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Número de árvores na floresta.
        max_depth : int ou None
            Profundidade máxima de cada árvore.
        random_state : int
            Semente para reprodutibilidade.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def build(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Instancia e treina o RandomForestClassifier.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Features de treino.
        y_train : array-like, shape (n_samples,)
            Labels binárias de surto (0/1).

        Returns
        -------
        self
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predições de classe (0 ou 1).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame build() primeiro.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades da classe positiva (surto=1).

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame build() primeiro.")
        return self.model.predict_proba(X)[:, 1]

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict = None,
        scoring_fn=None,
        cv_folds: list = None
    ) -> dict:
        """
        Busca em grade com validação cruzada temporal para encontrar os
        melhores hiperparâmetros do Random Forest.

        Parameters
        ----------
        X_train : array-like
            Features de treino completo.
        y_train : array-like
            Labels de treino completo.
        param_grid : dict, opcional
            Grade de hiperparâmetros. Padrão usa valores convencionais.
        scoring_fn : callable, opcional
            Função de métrica f(y_true, y_pred) → float (maior = melhor).
            Se None, usa acurácia simples.
        cv_folds : list, opcional
            Lista de folds gerada por validacao_cruzada_temporal().
            Se None, usa split temporal simples 80/20 interno.

        Returns
        -------
        dict
            Melhor configuração com chave 'params' e 'score'.
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [2, 3, 5, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }

        if scoring_fn is None:
            def scoring_fn(y_true, y_pred):
                return np.mean(y_true == y_pred)

        best_score = -np.inf
        best_params = None

        grid = list(ParameterGrid(param_grid))
        print(f"[RandomForestModel] Testando {len(grid)} configurações...")

        for params in grid:
            fold_scores = []

            if cv_folds is not None:
                for fold in cv_folds:
                    Xf_tr = X_train[fold["treino"]]
                    Xf_te = X_train[fold["teste"]]
                    yf_tr = y_train[fold["treino"]]
                    yf_te = y_train[fold["teste"]]

                    clf = RandomForestClassifier(
                        random_state=self.random_state, **params
                    )
                    clf.fit(Xf_tr, yf_tr)
                    y_pred = clf.predict(Xf_te)
                    fold_scores.append(scoring_fn(yf_te, y_pred))
            else:
                split = int(len(X_train) * 0.8)
                clf = RandomForestClassifier(
                    random_state=self.random_state, **params
                )
                clf.fit(X_train[:split], y_train[:split])
                y_pred = clf.predict(X_train[split:])
                fold_scores.append(scoring_fn(y_train[split:], y_pred))

            score = np.mean(fold_scores)
            if score > best_score:
                best_score = score
                best_params = params

        if best_params:
            self.n_estimators = best_params.get("n_estimators", self.n_estimators)
            self.max_depth = best_params.get("max_depth", self.max_depth)
            print(f"[RandomForestModel] Melhor config → {best_params}, score={best_score:.4f}")

        return {"params": best_params, "score": best_score}