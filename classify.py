"""
Este archivo contiene lo necesario para predecir una nueva transmisión.
"""

import pickle
import sys

import pandas as pd

from base_node import BaseNode


class Classify(BaseNode):
    """Clase para predecir el tipo de transmisión."""

    def __init__(self) -> None:
        super().__init__()
        self.model_name = sys.argv[1]
        self.paths = sys.argv[2:]

    def _read_model(self, filename: str):
        """Función para leer modelos de ml.

        Parameters
        ----------
            filename str:
                Nombre del archivo a leer.

        Returns
        -------
            Modelo previamente guardado.
        """
        filename = f"./models/{filename}.pkl"
        return pickle.load(open(filename, 'rb'))

    def _predict(self) -> None:
        """Principal función para predecir las nuevas transmisiones."""
        log_reg = self._read_model(self.model_name)
        vectorized = self._read_model("vectorized")
        for path in self.paths:
            file = open(path, "r", encoding="latin-1")
            transmission = file.read()
            df_trans = pd.DataFrame({"texto": [transmission]})
            df_trans = self._preprocessing(df_trans)
            df_trans = self._lemmatize(df_trans)
            df_trans = self._sentiment(df_trans)
            text = df_trans["texto"].values
            text_vectorized = vectorized.transform(text)
            columns = vectorized.get_feature_names()
            X = text_vectorized.toarray()
            df_2 = pd.DataFrame(X, columns=columns)
            df_2["SENTIMENT"] = df_trans['sentiment_score']
            print(f"{path}: {log_reg.predict(df_2.values)[0]}")

if __name__ == "__main__":
    Classify()._predict()
