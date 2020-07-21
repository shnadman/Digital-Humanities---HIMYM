# Import libraries
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from utils import *
from objects import *
import os
import json
import csv
from textblob import TextBlob # Create quick lambda functions to find the polarity and subjectivity of each routine
# URLs of transcripts in scope
import objects
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from geopy.geocoders import Nominatim
import itertools
from wordcloud import WordCloud
import gensim
import json

from nltk.corpus import stopwords

from textblob import TextBlob
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import nltk
nltk.download('wordnet')

geolocator = Nominatim(user_agent="HIMYM")

jar = './stanford-ner-tagger/stanford-ner.jar'
model = './stanford-ner-tagger/ner-model-english.ser.gz'

st = StanfordNERTagger(model, jar, encoding='utf8')


BASE_URL = 'https://transcripts.foreverdreaming.org/viewtopic.php?f=177&t='

urls = []

not_important = []


simpleDic ={'narrator': [], 'ted': [], 'lily': [], 'son':[], 'marshall':[], 'barney':[], 'robin': []}



characters = [Character('ted', ['narrator'], 'Male'), Character('lily', 'none', 'Female'),
              Character('marshall', 'marshal', 'Male'), Character('robin', 'none', 'Female'),
              Character('barney', 'none', 'Male')]



def get_character(name):
    name = name.split('(')[0].lower().strip()
    for c in characters:
        if re.search(fr'\b{name}\b', c.name.lower()):
            return c
        for alias in c.aliases:
            if re.search(fr'\b{name}\b', alias.lower()):
                return c

# Add line object to the line list of the given character
def add_line(character, season, ep, text):
    line = Line(season, ep, text)
    character.lines[str(season)].append(line)

def init_urls():
    for currEp in range(11505, 11713):
        currUrl = BASE_URL+str(currEp)
        urls.append(currUrl)
    # # # Actually request transcripts (takes a few minutes to run)
        [webscrape(u) for u in urls]



def webscrape(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")

    header = soup.find(lambda tag: tag.name == 'div' and
                                  tag.get('class') == ['boxheading']).text.strip('\n').strip('?')
    text = [p.text for p in soup.find(class_="postbody").find_all('p')]

    tokenized = header.split('x')
    season = tokenized[0]
    episode = tokenized[1].split(' - ')[0]

    text_file = open(f"transcripts/S{season}E{episode}.txt", "w+", encoding="utf-8")
    text_file.write(combine_text(text))
    text_file.close()
    print(header)
    print(url)
    return text



# Parse transcript of content of ep in season (go over characters names and lines and update character list

def parser(content, season, ep, locations ):
    tokenized_text = word_tokenize(content)
    classified_text = st.tag(tokenized_text)
    for word, tag in classified_text:
        if tag == 'LOCATION':
               locations.append(Location(season, ep, word))

    lines = content.splitlines()
    for l in lines:
        parts = l.split(":")
        if len(parts) >= 2 and len(parts[0]) < 20 and '[' not in parts[0]:
            # line of character
            character = get_character(parts[0])
            if character:
                add_line(character, season, ep, parts[1].strip())
            else:
                if parts[0] not in not_important:
                    not_important.append(parts[0])


def parse_series():
    locations = []
    for file in os.listdir("transcripts"):
        try:
            with open(f"transcripts/{file}", 'r') as episode:
               parser(episode.read(), int(file[2]), int(file[4:6]),locations)
        except Exception:
            with open(f"transcripts/{file}", 'r', encoding='utf8') as episode:
                parser(episode.read(), int(file[2]), int(file[4:6]), locations)
    firstRow=['Location', 'Season', 'Episode']
    with open('csvs/basic_locs.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(firstRow)

        for loc in locations:
                row = [loc.loc, loc.season, loc.ep]
                writer.writerow(row)


def count_words_per_season():
    for character in characters:
        for season in range(1, 10):
            character.season_counter[str(season)] = sum([l.wordCounter for l in character.lines[str(season)]])
        character.total_words = sum(character.season_counter.values())


def sa_per_season():
    for character in characters:
        for season in range(1, 10):
            season_lines = character.lines[str(season)]
            texts = []
            for l in season_lines:
                texts.append(l.text)

            corpus = combine_text(texts)

            character.polarity[str(season)] = TextBlob(corpus).polarity
            character.subjectivity[str(season)] = TextBlob(corpus).subjectivity



#Running this in one go may exceed the daily limits of requests
def geolocs_csv():
    try:
        with open('csvs/geolocs.csv', 'r') as geoFile, open('csvs/basic_locs.csv', 'r') as baseFile :
            first_row = ['Name', 'longitude', 'latitude']
            reader=csv.reader(baseFile, lineterminator='\n')
            writer=csv.writer(geoFile,lineterminator='\n')
            writer.writerow(first_row)
            for row in reader:
                loc=row[0]
                location = geolocator.geocode(loc)
                if location is not None:
                    rowToWrite=[loc,location.longitude, location.latitude]
                    writer.writerow(rowToWrite)
    except Exception:
        print("Error while opening file")
#


def export():
    with open('charcters.json', 'w') as output:
        json.dump(characters, output, default=lambda c: c.__dict__, indent=4) # To make it serialiazble



# [Name,Gender,House, number of words per season for each season, # of Words in total] csv creation
def create_polarity_csv():
    global characters
    first_row = ['Name','Season1', 'Season2', 'Season3', 'Season4', 'Season5', 'Season6', 'Season7',
                 'Season8', 'Season9']
    with open('csvs/polarity.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(first_row)
        for character in characters:
            row = [character.name, *character.polarity.values()]
            writer.writerow(row)

def create_subjectivity_csv():
    global characters
    first_row = ['Name','Season1', 'Season2', 'Season3', 'Season4', 'Season5', 'Season6', 'Season7',
                 'Season8', 'Season9']
    with open('csvs/subjectivity.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(first_row)
        for character in characters:
            row = [character.name, *character.subjectivity.values()]
            writer.writerow(row)


def create_sentiment_analysis_csv():
    create_polarity_csv()
    create_subjectivity_csv()


# [Name,Gender,House, number of words per season for each season, # of Words in total] csv creation
def create_main_csv():
    global characters
    first_row = ['Name', 'Gender', 'Season1', 'Season2', 'Season3', 'Season4', 'Season5', 'Season6', 'Season7',
                 'Season8', 'Season9', 'Total']
    with open('csvs/main.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(first_row)
        for character in characters:
            row = [character.name, character.gender, *character.season_counter.values(),
                   character.total_words]
            writer.writerow(row)


# [Name,Gender,House, Season number, # of Words] csv creation
def main_separately_csv():
    global characters
   # with open('character_list_results.json', 'r') as file:
   #     characters_tmp = json.load(file)
    first_row = ['Name', 'Gender', 'Season', 'Total']
    with open('csvs/main_seasons_separately.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(first_row)
        for character in characters:
            for key, season_counter in character.season_counter.items():
                row = [character.name, character.gender, key, season_counter]
                writer.writerow(row)


# [Name,Gender,# of Words] csv creation for each season
def create_csv_per_season():
    global characters
    first_row = ['Name', 'Gender', 'Words']
    for se in range(1, 10):
        with open(f'csvs/seasons/words_season_{str(se)}.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, lineterminator='\n')
            writer.writerow(first_row)
            for c in characters:
                row = [c.name, c.gender, c.season_counter[str(se)]]
                writer.writerow(row)


# [season number, # of words] csv creation for each character
def create_csv_per_character():
    global characters
    first_row = ['Season', 'Words']
    for c in characters:
        file_name = c.name
        with open(f'csvs/characters/{file_name}.csv', 'w') as csvFile:
            writer = csv.writer(csvFile, lineterminator='\n')
            writer.writerow(first_row)
            for se in range(1, 10):
                row = [se, c.season_counter[str(se)]]
                writer.writerow(row)


# [season number, # of episodes] csv creation
def season_episodes_csv():
    arr = [22, 22, 20, 24, 24, 24, 24, 24, 24]
    with open(f'csvs/episodes.csv', 'w') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(["season", "#episodes"])
        for i in range(1, 10):
            writer.writerow([i, arr[i - 1]])



def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))




# Tokenize and lemmatize
def preprocess1(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result



# words cloud algorithm including only nouns
def word_cloud(stop_words):
    wordcloudFemale = WordCloud(background_color="white", width=900, height=400, max_words=300, contour_width=6,
                          stopwords=stop_words)
    wordcloudMale = WordCloud(background_color="black", width=900, height=400, max_words=300, contour_width=6,
                          stopwords=stop_words)

    with open('charcters.json', 'r') as f:
        characters = json.load(f)
        female_lines = list(
            itertools.chain.from_iterable([itertools.chain.from_iterable(c["lines"].values()) for c in characters if c['gender'] == "Female"]))
        male_lines = list(
            itertools.chain.from_iterable([itertools.chain.from_iterable(c["lines"].values()) for c in characters if c['gender'] == "Male"]))
    # processed_lines = [TextBlob(line["text"]).noun_phrases for line in lines]
    processed_lines_female = [[word.lower() for (word, pos) in nltk.pos_tag(nltk.word_tokenize(line["text"])) if pos[0] == 'N']
                       for
                       line in female_lines]
    processed_lines_male = [
        [word.lower() for (word, pos) in nltk.pos_tag(nltk.word_tokenize(line["text"])) if pos[0] == 'N']
        for
        line in male_lines]
    wordcloudMale.generate(','.join(list(itertools.chain.from_iterable(processed_lines_male))))
    wordcloudMale.to_image().save('maleLines.png')
    wordcloudFemale.generate(','.join(list(itertools.chain.from_iterable(processed_lines_female))))
    wordcloudFemale.to_image().save('femaleLines.png')



simpleDic ={'ted': [], 'lily': [], 'marshall':[], 'barney':[], 'robin': []}
full_names = ['Ted Mosby', 'Lily', 'Marshall', 'Barney', 'Robin']

def combine_all():
    for c in characters:
        for lines in c.lines.values():
            for line in lines:
                simpleDic[c.name].append(line.text)

    data_combined = {key: [combine_text(value)] for (key, value) in simpleDic.items()}
    return data_combined


def combine_all_per_season(season):
    charDic = {'ted': [], 'lily': [], 'marshall': [], 'barney': [], 'robin': []}

    for c in characters:
        for lines in c.lines.values():
            for line in lines:
                if line.season==season:
                    charDic[c.name].append(line.text)

    data_combined = {key: [combine_text(value)] for (key, value) in charDic.items()}
    return data_combined


def generate_seasons_frames():
    framesArray = []
    for season in range(1,10):
        framesArray.append(combine_all_per_season(season))

    return framesArray

