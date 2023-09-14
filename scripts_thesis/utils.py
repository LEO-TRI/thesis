import nltk
import string
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def remove_proper_nouns(text):
    sentences = nltk.sent_tokenize(text)
    filtered_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        #words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
        tagged_words = nltk.pos_tag(words)
        filtered_words = [word for word, tag in tagged_words if tag != 'NNP' and tag != 'NNPS' and word not in string.punctuation and word not in stopwords]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)

    filtered_text = ' '.join(filtered_sentences)

    return filtered_text

def date_range(min_date, date, max_date):
    return (min_date <= date <= max_date)
date_range_vec = np.vectorize(date_range)

def update_prop(handle, orig):
    handle.update_from(orig)
    x,y = handle.get_data()
    handle.set_data([np.mean(x)]*2, [0, 2*y[0]])

def table_color(data:pd.Series, palette_min=145, palette_up=300, n=5):

    hex_colors = [mcolors.to_hex(color) for color in sns.diverging_palette(palette_min, palette_up, s=60, n=n)]
    hex_colors.reverse()

    bins = np.quantile(data.values, np.arange(0, 1.1, 0.2))
    vals = data.values

    cell_color = [hex_colors[0] if val <= bins[0] else hex_colors[1] if bins[0] < val < bins[1] else hex_colors[2] if bins[1] < val < bins[2] \
        else hex_colors[3] if bins[2] <= val <= bins[3] else hex_colors[4] for val in vals]

    return cell_color

def line_adder(h_coord=0.5, color="black", linestyle="-", *args):
    line = plt.Line2D([0.15,0.85], [h_coord, h_coord], transform=args.transFigure, color="black", linestyle="-")
    return args.add_artist(line)

def custom_combiner(feature, category):
    return str(category)
