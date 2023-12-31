import numpy as np
import pandas as pd
import string
import spacy
import spacy_fastlang
from spacy.language import Language
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords

class SpacyClean:
    """
    Cleaning class built around 2 Spacy pipelines.
    """

    def __init__(self):
        #Creating 2 spacy models. 1 for French, 1 for English.
        #fast_lang comes here and is integrated in the English model. Will be used to differentiate between languages

        self.nlp_en = spacy.load(('en_core_web_sm'))

        self.nlp_en.add_pipe("language_detector")
        self.nlp_en.remove_pipe('ner')

        #Creating the french model
        self.nlp_fr = spacy.load(('fr_core_news_sm'))
        self.nlp_fr.remove_pipe('ner')

    def preprocess_spacy(self, alpha: np.ndarray, batch_size: int=128) -> list:
        """
        Function using Spacy to lemmatize the text. Discriminates between french and english texts.

        Returns the lemmatized version of words when those words are NOUN, VERB and ADJ.

        Parameters
        ----------
        alpha : array_like
            A text column of a pd.DataFrame. Each cell must be a string.

        Returns
        -------
        docs : np.array
            An array of strings, each string being a text cleaned according to the method.

        mask: np.array
            An array of ints, each number being an indice. Used to track which texts weren't in French or English and drop them later.
        """
        alpha = [str(text) for text in alpha]

        for batch in tqdm((alpha[i:i + batch_size] for i in range(0, len(alpha), batch_size))):
            for doc in self.nlp_en.pipe(batch):
                tokens = []
                if doc._.language == "en":
                    for token in doc:
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            tokens.append(token.lemma_)
                    yield ' '.join(tokens)

                else:
                    for token in self.nlp_fr(doc.text):
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            tokens.append(token.lemma_)
                    yield ' '.join(tokens)

class CleanData:
    """
    A class centralising all the cleaning functions for text
    """

    def __init__(self):
        self.french_stopwords = stopwords.words("french")
        self.translator_p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self.translator_d = str.maketrans('', '', string.digits)

        #The class' methods, but in vectorized format for increased speed.
        #Can now be passed directly with the format clean_vec(array_like)
        self.clean_vec = np.vectorize(self.clean)
        self.clean_price_vec = np.vectorize(self.clean_price)


    def remove_proper_nouns(self, text: str) -> str:
        """
        Cleans the text.

        Remove punctuations and stopwords and proper nouns.

        Less accurate than SpacyClean.preprocess_spacy but a lot faster.

        Parameters
        ----------
        text : str
            A text to be cleaned, in string format

        Returns
        -------
        filtered_text : str
            A cleaned text, in string format
        """
        sentences = nltk.sent_tokenize(text)
        filtered_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence, language="french")
            tagged_words = nltk.pos_tag(words)
            filtered_words = [word for word, tag in tagged_words if tag != 'NNP' and tag != 'NNPS']
            filtered_words = [word for word in filtered_words if word not in string.punctuation and word not in self.french_stopwords]
            filtered_sentence = ' '.join(filtered_words)
            filtered_sentences.append(filtered_sentence)

        filtered_text = ' '.join(filtered_sentences)

        return filtered_text


    def clean(self, text: str) -> str:
        """
        Simple cleaning function. To be used with a .map or vectorized.

        Lowers text, remove whitespace, digits and punctuations

        Parameters
        ----------
        text : str
            A cell of a text column in string format

        Returns
        -------
        str
            A cell of a text column in string format, cleaned
        """
        text = text.strip()
        text = text.translate(self.translator_p)
        text = text.translate(self.translator_d)
        text = text.lower()
        return " ".join(text.split())


    def clean_price(self, text: str) -> float:
        """
        Function used to transform the price column from string to number.

        The function removes the dollar sign and rework the , vs . separating scheme.

        Parameters
        ----------
        text : str
            A cell of the column price, e.g. $100,000

        Returns
        -------
        float
            A float, e.g. 100000
        """
        translator_p = str.maketrans('', '', ",")
        text = text[1:]
        text = text.translate(translator_p)
        return float(text)
