########################### DL TEMPLATE ##############################
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers, models, Input
from tensorflow.keras.models import Sequential

import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize

from sklearn.model_selection import train_test_split

import gensim.downloader as api
word2vec_transfer = api.load("glove-wiki-gigaword-100")

from scripts_thesis.data import DataLoader

df = DataLoader.load_processed_data()

X = df.drop(columns=["license"])
y = df["license"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    train_size=0.8,
                                    random_state=42)


def embed_sentence_with_TF(word2vec: object, sentence: list[str]) -> list:
    """
    Function to embed words into vectors 

    Parameters
    ----------
    word2vec : object
        The model used to provide the embeddings
    sentence : list[str]
        The sentence to embed 

    Returns
    -------
    list
        An list of embeddings for each sentence
    """
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return embedded_sentence


def preprocess_dl_text(X: np.ndarray, max_len: int=200) -> tf.Tensor:
    """
    A function to centralise preprocessing text for a tf neural network

    Parameters
    ----------
    X : np.array
        A 1-dimension np.array of strings
    max_len: int
        The len of the outputed tensor. Will determine how much each text will be padded/truncated

    Returns
    -------
    tf.Tensor
        A full tensor of embeddings 
    """

    X = [word_tokenize(text) for text in X]
    X_embed = [embed_sentence_with_TF(word2vec_transfer, sentence) for sentence in X] 
    X_pad = tf.keras.utils.pad_sequences(X_embed, dtype='float32', padding='post', maxlen=max_len)

    return tf.convert_to_tensor(X_pad)


X_train = preprocess_dl_text()
X_test = preprocess_dl_text()

embedding_dims = 200 #Length of the token vectors
filters = 10 #number of filters in your Convnet
kernel_size = 3 # a window size of 3 tokens




nlp_input = Input(shape=(seq_length,), name='nlp_input')
emb = Embedding(output_dim=embedding_size, input_dim=100, input_length=seq_length)(nlp_input)
nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(emb)

meta_input = Input(shape=(10,), name='meta_input')

x = concatenate([nlp_out, meta_input])
x = Dense(classifier_neurons, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[nlp_input , meta_input], outputs=[x])



model = Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=False)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh")))
model.add(layers.Dense(130, activation="relu"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(80, activation="relu"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(y_train.shape[1], activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                    tf.keras.metrics.Precision()]
            )

epochs = 500

es = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    batch_size = 16,
                    #verbose = 0,
                    epochs=epochs,
                    callbacks=[es]
                    )


