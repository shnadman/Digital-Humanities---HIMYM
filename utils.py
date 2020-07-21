import requests
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import re
import string
import numpy as np
import math
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from objects import Line,Character




# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
    print(url)
    return text


# We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = '\n '.join(list_of_text)
    return combined_text

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    #text = re.sub('\s+', ' ', text)
    #text = re.sub("\'", "", text)

    return text


def create_document_term_matrix2(data,stopWords):
    """Given a panda frame, make a DTM out of it and save it as a pickle file"""

    cv = CountVectorizer(stop_words=stopWords)
    data_cv = cv.fit_transform(data.transcript)
    data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.index = data.index
    pickle.dump(cv, open("cv_stop.pkl", "wb"))
    data_stop.to_pickle("dtm_stop.pkl")

    return data_stop


def dict_to_panda(dict, columns):
    """Usage: dictToPanda(dict,['col1', 'col2'])"""
    pd.set_option('max_colwidth',150)
    data_df = pd.DataFrame.from_dict(dict).transpose()
    data_df.columns = columns
    data_df = data_df.sort_index()
    return data_df



def split_text(text, n=10):
        '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

        # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
        length = len(text)
        size = math.floor(length / n)
        start = np.arange(0, length, size)

        # Pull out equally sized pieces of text and put it into a list
        split_list = []
        for piece in range(n):
            split_list.append(text[start[piece]:start[piece] + size])
        return split_list

# Let's create a function to pull out nouns from a string of text
def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)


