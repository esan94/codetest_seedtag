"""
Este archivo contiene lo necesario para ejecutar el entrenamiento del modelo de clasificación
de transmisiones.
"""

import pickle
import sys

import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression

from base_node import BaseNode

nltk.download('words')
nltk.download('vader_lexicon')


class Train(BaseNode):
    """Clase principal para entrenar un sistema de clasificacion de transmisiones."""

    def __init__(self) -> None:
        super().__init__()
        self.path = sys.argv[1]
        self.model_file = "./models/LogReg.pkl"

    def _fit(self, dataframe: pd.DataFrame) -> LogisticRegression:
        """Este método entrena una regresión logística.

        Parameters
        ----------
            dataframe pd.DataFrame:
                Datos de entrenamiento

        Returns
        -------
            LogisticRegression:
                Modelo de regresión logística entrenado.
        """

        model = LogisticRegression(C=1, class_weight="balanced")
        X_train = dataframe.select_dtypes(float).values
        Y_train = dataframe.select_dtypes(object).values
        model.fit(X_train, Y_train)
        return model

    def _save_model(self, trained_model: LogisticRegression) -> None:
        """Método para guardar el modelo entrenado.

        Parameters
        ----------
            trained_model LogisticRegression:
                Modelo de regresión logística entrenado.

        Returns
        -------
            None
        """

        pickle.dump(trained_model, open(self.model_file, 'wb'))

    def _train(self) -> None:
        """Método principal de la clase."""
        df = self._read(self.path)
        df = self._preprocessing(df)
        df = self._lemmatize(df)
        df = self._sentiment(df)
        df = self._tfidf(df)
        trained_model = self._fit(df)
        self._save_model(trained_model)

if __name__ == "__main__":
    Train()._train()
    print("Modelo entrenado.")
