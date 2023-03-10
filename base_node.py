import os
import pickle
from typing import Dict, Optional

import nltk.sentiment.vader as vd
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseNode():
    """Clase comun con funciones generales."""

    def __init__(self) -> None:
        self.stop_words_en = stopwords.words("english")
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.lemmatizer = WordNetLemmatizer()
        self.pos_map_lemma = {
                                "JJ": "a", "JJR": "a", "JJS": "a", "NN": "n", "NNS": "n", "RB": "r",
                                "RBR": "r", "RBS": "r", "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v",
                                "VBP": "v", "VBZ": "v"
                                }

    @staticmethod
    def remove_extra_whitespace(string: str) -> str:
        """Esta función toma un string y le elimina los espacio extra que puedan tener las palabras.

        Parameters
        ----------
            string str:
                Frase sobre la que se quiere aplicar la función.

        Returns
        -------
            str:
                Frase sin espacios blancos extra.
        """

        words = string.split(" ")
        return " ".join([word.strip() for word in words if word != ""])

    def _read(self, path: str, files: Optional[Dict] = None) -> pd.DataFrame:
        """Esta función se usa para leer archivos de una ruta dada.

        Esta función busca subcarpetas dentros de una ruta dada y lee todos los archivos que se
        encuentren por debajo de dicha subcarpeta para crear un DataFrame con la estructura de
        categoría y texto. Si se le pasa un diccionario por el argumento files, solo selecciona
        para leer la lista de archivos que se le hayan pasado, por categoría, al argumento.

        Parameters
        ----------
            path str:
                Ruta donde se encuentran los subdirectorios con los archivos por categorías.

            files Optional[Dict]:
                Diccionario donde las claves son las categorías y los valores una lista de archivos
                que se quieren leer. Por defecto esta variable se encuentra inicializada a None.

        Returns
        -------
            pd.DataFrame:
                DataFrame con todos los datos leídos y con columnas target y texto.
        """
        data = []
        list_cat = os.listdir(path)

        for cat in list_cat:
            if files is None:
                list_files = os.listdir(f"{path}/{cat}")
                for file in list_files:
                    ruta_completa = f"{path}/{cat}/{file}"
                    archivo = open(ruta_completa, "r", encoding="latin-1")
                    data.append({"TARGET": cat, "texto": archivo.read()})
            else:
                for file in files[cat]:
                    ruta_completa = f"{path}/{cat}/{file}"
                    archivo = open(ruta_completa, "r", encoding="latin-1")
                    data.append({"TARGET": cat, "texto": archivo.read()})

        return pd.DataFrame(data)

    def _lemma(self, text: str, lemmatizer: WordNetLemmatizer, pos_lemma: Dict) -> str:
        """Realiza la lematización.

        Realiza la lematización teniendo en cuenta el tipo de palabra. Si alguna palabra no se
        le encuentra un tipo se la considera nombre.

        Parameters
        ----------
            text: str
                Transmisión sobre la que realizar la operación.

            lemmatizer: WordNetLemmatizer
                    Instancia del objeto WordNetLemmatizer.

            pos_lemma: dict
                Mapeo de los tipos de palabras.

        Returns
        -------
            str:
                Transmisión lematizada.
        """
        return " ".join([lemmatizer.lemmatize(token[0],
                     pos=pos_lemma.get(token[1], "n"))
                     for token in pos_tag(word_tokenize(text))])

    def _lemmatize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Aplica la lematización sobre todos los datos.

        Parameters
        ----------
            dataframe pd.DataFrame:
                Datos a lematizar.

        Returns
        -------
            pd.DataFrame:
                Datos lematizados.
        """
        dataframe["texto"] = dataframe["texto"].apply(lambda _: self._lemma(_, self.lemmatizer, self.pos_map_lemma))
        return dataframe

    def _tfidf(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Calcula el total frecuency per inverse document frecuency.

        Parameters
        ----------
            dataframe pd.DataFrame:
                Datos a lematizar.

        Returns
        -------
            pd.DataFrame:
                Datos lematizados.
        """
        vectorizer = TfidfVectorizer(stop_words=["english"], max_features=27000)
        train_text = dataframe["texto"].values
        text_vectorized = vectorizer.fit_transform(train_text)
        pickle.dump(vectorizer, open("./models/vectorized.pkl", 'wb'))
        columns = vectorizer.get_feature_names()
        X_train = text_vectorized.toarray()
        df_2_train = pd.DataFrame(X_train, columns=columns)
        df_2_train["SENTIMENT"] = dataframe['sentiment_score']
        df_2_train["TARGET"] = dataframe['TARGET']
        return df_2_train

    def _sentiment(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Analiza los sentimientos de una transmisión.

        Parameters
        ----------
            dataframe pd.DataFrame:
                Datos a analizar.

        Returns
        -------
            pd.DataFrame:
                Datos con el análisis de sentimiento.
        """
        sia = vd.SentimentIntensityAnalyzer()
        dataframe['sentiment_score'] = dataframe["texto"].apply(lambda x: sia.polarity_scores(x)['compound'])
        return dataframe

    def _preprocessing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Preprocesado del texto.

        Parameters
        ----------
            dataframe pd.DataFrame:
                Datos a preprocesar.

        Returns
        -------
            pd.DataFrame:
                Datos preprocesados.
        """
        dataframe["texto"] = dataframe["texto"].str.lower()
        dataframe["texto"] = dataframe["texto"].str.replace("\S*@\S*\s?", "-")
        dataframe["texto"] = dataframe["texto"].str.replace("\n", " ")
        dataframe["texto"] = dataframe["texto"].str.replace("\t", " ")

        text = []
        for words in dataframe["texto"].str.split(" ").values:
            row = []
            for word in words:
                if word not in self.stop_words_en:
                    row.append(word)
            text.append(" ".join(row))
        dataframe["texto"] = text

        punctuation = dict.fromkeys(self.punctuation, " ")
        dataframe["texto"] = dataframe["texto"].apply(lambda syn: syn.translate(str.maketrans(punctuation)))

        text = []
        for words in dataframe["texto"].str.split(" ").values:
            row = []
            for word in words:
                if word not in self.stop_words_en:
                    row.append(word)
            text.append(" ".join(row))
        dataframe["texto"] = text

        dataframe["texto"] = dataframe["texto"].str.replace("\d+", " ")
        dataframe["texto"] = dataframe["texto"].str.replace("\b\w\b", " ")
        dataframe["texto"] = dataframe["texto"].apply(lambda _: self.remove_extra_whitespace(_))
        dataframe["texto"] = dataframe["texto"].str.replace(r"(\w)\1*", r"\1")

        return dataframe
