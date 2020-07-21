
# LDA
import gensim
from gensim import corpora, models, similarities
import warnings
import spacy
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
import nlp
from nltk.corpus import stopwords
import pickle

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import IPython
import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])




def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,trigram_mod,bigram_mod):

    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def topic_model(data_df):

    data = data_df.transcript.values.tolist()
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)

    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #tokenized_clean = data_df['transcript'].map(cleaning)

    texts = data_lemmatized
    # Creating term dictionary of corpus, where each unique term is assigned an index.
    dictionary = corpora.Dictionary(data_lemmatized)
    # Filter terms which occurs in less than 1 review and more than 80% of the reviews.
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    # convert the dictionary to a bag of words corpus
    corpus = [dictionary.doc2bow(tokens) for tokens in texts]

    warnings.simplefilter("ignore", DeprecationWarning)


    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=5,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=100,
                                               alpha='auto',
                                               per_word_topics=True)


    ldamodel.save('model_combined.gensim')


    topics = ldamodel.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    # Didn't work for us, but worth a try - Visualizes topic modeling results
    # vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    # pyLDAvis.display(vis)

    return data_lemmatized


