import pandas as pd
from collections import Counter
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stemmer = PorterStemmer()

def generator_preper(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.
    """
    # remove_digits and special chars
    text = re.sub("[^a-zA-Z ]", "", text)
    return text



def initial_clean(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.
    """
    # remove_digits and special chars
    text = re.sub("[^a-zA-Z ]", "", text)

    text = text.lower()  # lower case text
    text = nltk.word_tokenize(text)
    return text


stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])


def stem_words(text):
    """
    Function to stem words
    """
    # try:
    text = [stemmer.stem(word) for word in text]
    text = [word for word in text if len(word) > 2]  # no single letter words
    # except IndexError:
    #    pass
    return text


def remove_stop_words(text):
    return [word for word in text if word not in stop_words]


def remove_non_english_words(text):
    filtered_text = []

    for token in text:

        if len(token) == 1:
            continue
        elif token in stop_words:
            continue
        elif not wordnet.synsets(token):
            # Not an English Word
            continue
        else:
            # English Word
            filtered_text.append(token)
    return filtered_text


def get_stop_words(data_dtm2):

    # Find the top 30 words said by each character
    top_dict = {}
    for c in data_dtm2.columns:
        top = data_dtm2[c].sort_values(ascending=False).head(30)
        top_dict[c]= list(zip(top.index, top.values))

    # Print the top 15 words said by each character
    for character, top_words in top_dict.items():
        print(character)
        print(', '.join([word for word, count in top_words[0:14]]))
        print('---')

    # Let's first pull out the top 30 words for each comedian
    words = []
    for character in data_dtm2.columns:
        top = [word for (word, count) in top_dict[character]]
        for t in top:
            words.append(t)

    # Let's aggregate this list and identify the most common words along with how many routines they occur in
    Counter(words).most_common()

    # If more than half of the characters have it as a top word, exclude it from the list, count = number of characters in the show
    add_stop_words = [word for word, count in Counter(words).most_common() if count > 3]

    return add_stop_words


def plot_unique_words(data,full_names):
    """ Find the number of unique words that each character uses and plots them"""
    # Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
    unique_list = []
    for character in data.columns:
        uniques = data[character].to_numpy().nonzero()[0].size
        unique_list.append(uniques)

    data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['character', 'unique_words'])
    print(data_words)
    data_unique_sort = data_words.sort_values(by='unique_words')
    y_pos = np.arange(len(data_words))

    plt.subplot(1, 2, 1)
    plt.barh(y_pos, data_unique_sort.unique_words, align='center')
    plt.yticks(y_pos, data_unique_sort.character)
    plt.title('Number of Unique Words', fontsize=20)

    plt.tight_layout()
    plt.show()


def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(remove_non_english_words(initial_clean(text))))