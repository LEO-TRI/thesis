import numpy as np
import nltk
import string
from nltk.corpus import stopwords
import spacy
import spacy_fastlang
from spacy.language import Language
from tqdm import tqdm

nlp = spacy.load(('en_core_web_sm'))
nlp.add_pipe("language_detector") #fast_lang comes here

french_stopwords = stopwords.words("french")
translator_p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
translator_d = str.maketrans('', '', string.digits)

def remove_proper_nouns(text:str) -> str:
    sentences = nltk.sent_tokenize(text)
    filtered_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence, language="french")
        tagged_words = nltk.pos_tag(words)
        filtered_words = [word for word, tag in tagged_words if tag != 'NNP' and tag != 'NNPS']
        filtered_words = [word for word in filtered_words if word not in string.punctuation and word not in french_stopwords]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)

    filtered_text = ' '.join(filtered_sentences)

    return filtered_text
remove_proper_nouns_vec = np.vectorize(remove_proper_nouns)

def clean(text):
    text = text.strip()
    text = text.translate(translator_p)
    text = text.translate(translator_d)
    text = text.lower()
    return " ".join(text.split())
clean_vec = np.vectorize(clean)

def clean_price(text:str) -> float:
    translator_p = str.maketrans('', '', ",")
    text = text[1:]
    text = text.translate(translator_p)
    return float(text)
clean_price_vec = np.vectorize(clean_price)

def preprocess_spacy(alpha: list[str]) -> list[str]:

    docs = list()
    mask = list()
    alpha = [str(text) for text in alpha]
    for ind, doc in tqdm(enumerate(nlp.pipe(alpha, batch_size = 128))):
        if doc._.language == "en":
            tokens = list()
            for token in doc:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    tokens.append(token.lemma_)
            docs.append(' '.join(tokens))
            mask.append(ind)

    return np.array(docs), mask
