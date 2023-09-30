########################### DL TEMPLATE ##############################
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import gensim.downloader as api
word2vec_transfer = api.load("glove-wiki-gigaword-100")

from scripts_thesis.data import DataLoader


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

#TODO
X_train = preprocess_dl_text()
X_test = preprocess_dl_text()

#emb = Embedding(output_dim=embedding_size, input_dim=100, input_length=seq_length)(nlp_input)
#nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(nlp_input)

class NeuralModel:

    def __init__(self, embedding_dims: int= 200, num_tags: int=20) -> None:
        self.embedding_dims = embedding_dims
        self.num_tags = num_tags

    def build_model(self) -> tf.keras.Model:
        """
        Function to build a double input neural network model

        Usually will rarely be called alone but will be used in compile_model()

        Returns
        -------
        tf.keras.Model
            The built model. Still needs to be compiled.
        """
        vocab_size = 5000
        max_len = 200

        vectorize_layer = tf.keras.layers.TextVectorization(
            ngrams = 2,
            max_tokens=self.embedding_dims,
            output_mode='int',
            output_sequence_length=max_len
            )
        vectorize_layer.adapt(X_train)

        cat_input = tf.keras.Input(shape=(self.num_tags,), name="tags")
        num_input = tf.keras.Input(shape=(10,), name='num_input') #TODO
        nlp_input = tf.keras.Input(shape=(None,), name='nlp_input')

        nlp_vec = vectorize_layer(nlp_input)
        nlp_features = tf.keras.layers.Embedding(self.embedding_dims+ 1, 64)(nlp_vec)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", dropout=0.3, return_sequences=True))(nlp_features)
        nlp_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=False))(x)

        x = tf.keras.layers.Concatenate([nlp_out, num_input, cat_input])

        x = tf.keras.layers.Dense(130, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        x = tf.keras.layers.Dense(80, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        pred = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=[nlp_input , num_input, cat_input], outputs=[pred])

        print(tf.keras.utils.plot_model(model, show_shapes=True))

        return model

    def compile_model(self, loss: str="binary_crossentropy", optimizer: str="adam") -> tf.keras.Model:
        """
        Function to compile a model.

        Uses build_model() to load a (not-compiled) model

        Usually will rarely be called alone but will be used in train_model()

        Parameters
        ----------
        loss : str, optional
            The loss function, by default "binary_crossentropy"
        optimizer : str, optional
            The optimizer, by default "adam"

        Returns
        -------
        tf.keras.Model
            The compiled model
        """
        model = self.build_model()
        model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision()])

        return model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int=500, patience: int=20, batch_size: int=16, validation_split: float=0.2) -> tf.keras.Model:
        """
        Function to centralise the methods required to train the neural network.

        Uses compile_model() to load the compiled model

        Parameters
        ----------
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
        model = self.compile_model()
        es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                    validation_split=validation_split,
                    batch_size = batch_size,
                    #verbose = 0,
                    epochs=epochs,
                    callbacks=[es]
                    )

        return history
