########################### DL TEMPLATE ##############################
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from scripts_thesis.TF_one_hot_encoder import OneHotEncodingLayer
from scripts_thesis.utils import ohe

import gensim.downloader as api
word2vec_transfer = api.load("glove-wiki-gigaword-100")

def embed_sentence_with_TF(word2vec: object, sentence: list[str]) -> list:
    """
    Function to embed words into vectors

    Parameters
    ----------
    word2vec : object
        The model used to provide the embeddings
    sentence : list[str]
        The sentence to embed represented as a list of words

    Returns
    -------
    list
        An list of embeddings for each sentence. Has dimensions (len(sentence) , len(word2vec_model))
    """
    return [word2vec[word] for word in sentence if word in word2vec]


def preprocess_dl_text(X: np.ndarray, max_len: int=200) -> tf.Tensor:
    """
    A function to centralise preprocessing text for a tf neural network

    Parameters
    ----------
    X : np.ndarray
        A 1-dimension np.array of strings
    max_len: int
        The len of the outputed tensor. Will determine how much each text will be padded/truncated

    Returns
    -------
    tf.Tensor
        A full tensor of embeddings
    """

    X = [tf.strings.split(text, " ").numpy() for text in X]
    X_embed = [embed_sentence_with_TF(word2vec_transfer, sentence) for sentence in X]
    X_pad = tf.keras.utils.pad_sequences(X_embed, dtype='float32', padding='post', maxlen=max_len)

    return tf.convert_to_tensor(X_pad)

#X_train = preprocess_dl_text()
#X_test = preprocess_dl_text()

#emb = Embedding(output_dim=embedding_size, input_dim=100, input_length=seq_length)(nlp_input)
#nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(nlp_input)

class NeuralModel:

    def __init__(self, embedding_dims: int= 200, num_tags: int=20,
                 nlp_cols: str=None, cat_cols: str=None, numeric_cols: list[str]=None) -> None:
        self.embedding_dims = embedding_dims
        self.num_tags = num_tags
        self.nlp_cols = nlp_cols
        self.cat_cols = cat_cols
        self.numeric_cols = numeric_cols

    def build_model(self, X_train: pd.DataFrame, y_train : pd.Series) -> tf.keras.Model:
        """
        Function to build a double input neural network model

        Usually will rarely be called alone but will be used in compile_model()

        Returns
        -------
        tf.keras.Model
            The built model. Still needs to be compiled.
        """
        self.nlp_cols = "description"
        self.cat_cols = "neighbourhood_cleansed"
        self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        self.neg, self.pos = np.bincount(y_train)

        vectorize_layer = tf.keras.layers.TextVectorization(
            ngrams = 2,
            max_tokens=200,
            output_mode='int',
            output_sequence_length=self.embedding_dims,
            pad_to_max_tokens=True
            )
        vectorize_layer.adapt(X_train.loc[:, "host_about"].values)

        num_input = tf.keras.Input(shape=(10,), name='num_input')
        cat_input = tf.keras.Input(shape=(1, ), name="cat_input", dtype=tf.int64)
        nlp_input = tf.keras.Input(shape=(1, ), name='nlp_input')

        cat_output =tf.keras.layers.CategoryEncoding(num_tokens=20, output_mode="one_hot")(cat_input)

        nlp_vec = vectorize_layer(nlp_input)
        nlp_features = tf.keras.layers.Embedding(200+ 1, 64)(nlp_vec)
        nlp_features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh",
                                                                dropout=0.3, return_sequences=True))(nlp_features)
        nlp_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh",
                                                                        return_sequences=False))(nlp_features)

        x = tf.keras.layers.Concatenate(axis=1)([num_input, cat_output, nlp_out])
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        pred = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=[num_input, cat_input], outputs=[pred])

        # Compile the model
        model.compile(loss="binary_crossentropy", optimizer="adam",
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.AUC()
                    ])
        return model

    def compile_model(self, X_train: pd.DataFrame, loss: str="binary_crossentropy",
                      optimizer: str="adam") -> tf.keras.Model:
        """
        Function to compile a model.

        Uses build_model() to load a (not-compiled) model

        Usually will rarely be called alone but will be used in train_model()

        Parameters
        ----------
        X_train: pd.DataFrame
            The dataframe of data to train one, required to create the vectorizer vocabulary
        loss : str, optional
            The loss function, by default "binary_crossentropy"
        optimizer : str, optional
            The optimizer, by default "adam"

        Returns
        -------
        tf.keras.Model
            The compiled model
        """
        model = self.build_model(X_train=X_train)
        model.compile(loss="binary_crossentropy", optimizer="adam",
                    metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision()])

        return model

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, epochs: int=500,
                    patience: int=20, batch_size: int=16, validation_split: float=0.2) -> tf.keras.Model:
        """
        Function to centralise the methods required to train the neural network.

        Uses compile_model() to load the compiled model

        Parameters
        ----------
        X_train: pd.DataFrame:
            The training set of features. Will be converted to tensor for fitting
        y_train: pd.Series:
            The training set of targets. Will be converted to tensor for fitting
        epochs : int, optional
            The max number of epochs, by default 500
        patience : int, optional
            The number of epochs before early stopping, by default 20
        batch_size : int, optional
            The batch size, by default 16
        validation_split: float, optional
            The validation split for the train data, by default 0.2

        Returns
        -------
        tf.keras.Model
            A trained model with weights and history
        """
        model = self.compile_model(X_train)
        es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

        weight_for_0 = (1 / self.neg) * ((self.neg+self.pos) / 2.0)
        weight_for_1 = (1 / self.pos) * ((self.neg+self.pos) / 2.0)
        class_weight = {0:weight_for_0, 1:weight_for_1}

        cat_data = ohe(X_train.loc[:, self.cat_cols])

        num_data = tf.convert_to_tensor(X_train.loc[:, self.numeric_cols].values)
        cat_data = tf.convert_to_tensor(X_train.loc[:, self.cat_cols].values)
        nlp_data = tf.convert_to_tensor(X_train.loc[:, self.nlp_cols].values)

        y_train = tf.convert_to_tensor(y_train.values)

        history = model.fit(dict(nlp_input=nlp_data, num_input=num_data, cat_input=cat_data),
                            y_train,
                            class_weight=class_weight,
                            validation_split=validation_split,
                            batch_size = batch_size,
                            #verbose = 0,
                            epochs=epochs,
                            callbacks=[es]
                            )

        return history
