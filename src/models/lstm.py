import numpy as np
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K
from keras_tuner import HyperModel, RandomSearch


class LSTMMultitaskHyperModel(HyperModel):
    """
    LSTM unidirecional multitarefa com saídas de classificação e regressão.
    """

    def __init__(self, window_size: int, num_features: int, pred_steps: int = 1):
        """
        Parameters
        ----------
        window_size : int
            Tamanho da janela temporal (número de passos de entrada).
        num_features : int
            Número de features por passo de tempo.
        pred_steps : int
            Número de passos futuros a prever na saída de regressão.
        """
        self.window_size = window_size
        self.num_features = num_features
        self.pred_steps = pred_steps

    def build(self, hp):
        K.clear_session()

        lstm_units   = hp.Int("lstm_units",   min_value=128, max_value=512, step=32)
        dense_units  = hp.Int("dense_units",  min_value=128, max_value=512, step=32)
        dropout_rate = hp.Float("dropout_rate", min_value=0.05, max_value=0.5, step=0.05)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        inp = layers.Input(shape=(self.window_size, self.num_features))
        x = layers.LSTM(units=lstm_units)(inp)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

        out_class = layers.Dense(1, activation="sigmoid", name="classification")(x)
        out_reg   = layers.Dense(self.pred_steps, activation="linear", name="regression")(x)

        model = models.Model(inputs=inp, outputs=[out_class, out_reg])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                "classification": "binary_crossentropy",
                "regression": "mean_squared_error"
            },
            metrics={
                "classification": "accuracy",
                "regression": "mae"
            }
        )
        return model


class LSTMBidirectionalMultitaskHyperModel(HyperModel):
    """
    LSTM bidirecional multitarefa com saídas de classificação e regressão.
    Recomendada como arquitetura principal conforme Farias, Silva e Araújo (2025).
    """

    def __init__(self, window_size: int, num_features: int, pred_steps: int = 1):
        """
        Parameters
        ----------
        window_size : int
            Tamanho da janela temporal.
        num_features : int
            Número de features por passo de tempo.
        pred_steps : int
            Passos futuros na saída de regressão.
        """
        self.window_size = window_size
        self.num_features = num_features
        self.pred_steps = pred_steps

    def build(self, hp):
        K.clear_session()

        lstm_units   = hp.Int("lstm_units",   min_value=128, max_value=512, step=32)
        dense_units  = hp.Int("dense_units",  min_value=128, max_value=512, step=32)
        dropout_rate = hp.Float("dropout_rate", min_value=0.05, max_value=0.5, step=0.05)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        inp = layers.Input(shape=(self.window_size, self.num_features))
        x = layers.Bidirectional(layers.LSTM(units=lstm_units))(inp)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

        out_class = layers.Dense(1, activation="sigmoid", name="classification")(x)
        out_reg   = layers.Dense(self.pred_steps, activation="linear", name="regression")(x)

        model = models.Model(inputs=inp, outputs=[out_class, out_reg])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                "classification": "binary_crossentropy",
                "regression": "mean_squared_error"
            },
            metrics={
                "classification": "accuracy",
                "regression": "mae"
            }
        )
        return model


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

class LSTMHyperparameterTuner:
    """
    Realiza busca aleatória de hiperparâmetros (RandomSearch via Keras Tuner)
    para qualquer HyperModel LSTM multitarefa.

    O tuner salva o melhor modelo em disco e o retorna para uso no notebook.
    """

    def __init__(
        self,
        hypermodel,
        x_train: np.ndarray,
        y_train_class: np.ndarray,
        y_train_regress: np.ndarray,
        seed: int = 42,
        epochs_tuning: int = 50,
        max_trials: int = 30,
        project_name: str = "dengue_lstm_tuning"
    ):
        """
        Parameters
        ----------
        hypermodel : HyperModel
            Instância de LSTMMultitaskHyperModel ou
            LSTMBidirectionalMultitaskHyperModel.
        x_train : np.ndarray, shape (n, window_size, num_features)
            Dados de entrada para treino/validação interna do tuner.
        y_train_class : np.ndarray, shape (n,)
            Labels de classificação (surto 0/1).
        y_train_regress : np.ndarray, shape (n,)
            Valores normalizados de casos para regressão.
        seed : int
            Semente de reprodutibilidade.
        epochs_tuning : int
            Épocas máximas por trial.
        max_trials : int
            Número de combinações testadas pelo RandomSearch.
        project_name : str
            Nome do diretório de saída do Keras Tuner.
        """
        self.hypermodel = hypermodel
        self.x_train = x_train
        self.y_train_class = y_train_class
        self.y_train_regress = y_train_regress
        self.seed = seed
        self.epochs_tuning = epochs_tuning
        self.max_trials = max_trials
        self.project_name = project_name

    def tune(self, save_path: str = "best_lstm_model.keras"):
        """
        Executa a busca de hiperparâmetros e retorna o melhor modelo salvo.

        O split interno usa 80 % para treino e 20 % para validação,
        respeitando a ordem temporal dos dados.

        Parameters
        ----------
        save_path : str
            Caminho para salvar o melhor modelo encontrado.

        Returns
        -------
        keras.Model
            Melhor modelo treinado com os hiperparâmetros otimizados.
        """
        split_idx = int(len(self.x_train) * 0.8)

        X_tr, X_val = self.x_train[:split_idx], self.x_train[split_idx:]
        y_cl_tr, y_cl_val = self.y_train_class[:split_idx], self.y_train_class[split_idx:]
        y_rg_tr, y_rg_val = self.y_train_regress[:split_idx], self.y_train_regress[split_idx:]

        tuner = RandomSearch(
            self.hypermodel,
            objective="val_loss",
            max_trials=self.max_trials,
            seed=self.seed,
            executions_per_trial=1,
            directory="hyperparam_tuning",
            project_name=self.project_name,
            overwrite=True
        )

        cb_early = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        cb_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
        )

        tuner.search(
            X_tr,
            {"classification": y_cl_tr, "regression": y_rg_tr},
            validation_data=(X_val, {"classification": y_cl_val, "regression": y_rg_val}),
            epochs=self.epochs_tuning,
            batch_size=32,
            callbacks=[cb_early, cb_lr],
            verbose=1
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save(save_path)
        print(f"[LSTMHyperparameterTuner] Melhor modelo salvo em: {save_path}")

        return best_model