# Import libraries

# URLs of transcripts in scope
from textGeneration import *
from webscarping import *
from eda import *
from topic_modeling import topic_model


stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people','guy','girl','thing','something','anything','nothing'])


init_urls()
parse_series()
count_words_per_season()
sa_per_season()
geolocs_csv()
export()
season_episodes_csv()
create_csv_per_character()
create_csv_per_season()
main_separately_csv()
create_main_csv()
create_sentiment_analysis_csv()




#-------------------------------------------------------------TOPIC MODELING----------------------------------------------------

# For each season seperately
frames = generate_seasons_frames()

for season,frame in enumerate(frames):

    df = dict_to_panda(frame, ['transcript'])

    df['full_name'] = full_names
    print(f'Topic model for season {season}:\n')
    topic_model(df)

# For every season combined
data_combined = combine_all()

data_df = dict_to_panda(data_combined, ['transcript'])
data_df['full_name'] = full_names
print(f'Topic model for the entire series:\n')
topic_model(data_df)

#-------------------------------------------------------------VOCABULARY----------------------------------------------------

base_clean = lambda x: clean_text(x)

data_clean = pd.DataFrame(data_df.transcript.apply(base_clean))

dtm1 = create_document_term_matrix2(data_clean, stop_words).transpose()
more_stop_words = get_stop_words(dtm1)
stop_words.extend(more_stop_words)
dtm2 = create_document_term_matrix2(data_clean, stop_words).transpose()
plot_unique_words(dtm2, full_names)  # Plotting a graph to see how big the vocabulary is of each individual

#---------------------------------------------------------WORD CLOUDS--------------------------------------------------------------

word_cloud(stop_words)

#---------------------------------------------------------TEXT GENERATION--------------------------------------------------------------


prep_for_generation = lambda x: generator_preper(x)

prepped = pd.DataFrame(data_df.transcript.apply(prep_for_generation))  # data after initial cleaning

print(prepped)
chain = markov_chain('barney',prepped)
print(generate_sentence(chain, 20))
