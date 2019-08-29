from GeniusArtistDataCollect import GeniusArtistDataCollect
from analyze import get_lexical_richness, get_sentiment_analysis, get_topic_choices, get_topic_analysis
import pandas as pd
import warnings
import os
import keyring

warnings.filterwarnings('ignore')


def main(artist_name, album_list):

    if os.path.isfile('{}.csv'.format(artist_name)):

        songs_df = pd.DataFrame.from_csv('{}.csv'.format(artist_name))

    else:

        client_access_token = keyring.get_password('api.genius.com', 'Genius Client Access Token')

        g = GeniusArtistDataCollect(client_access_token, artist_name, album_list)
        songs_df = g.get_artist_songs()

        songs_df.to_csv('{}.csv'.format(artist_name))

    get_lexical_richness(songs_df, artist_name, album_list)
    get_sentiment_analysis(songs_df, artist_name, album_list)

    more_stop_words = ['like', 'oh', 'yeah', 'em', 'get', 'got', 'could', 'cause', 'okay',
                       'well', 'let', 'ya', 'yo', 'said', 'wanna', 'need', 'goes', 'see', 'gotta']

    topics_tuple = get_topic_choices(songs_df, more_stop_words)
    topic_choices = topics_tuple[0]
    topics_matrix = topics_tuple[1]
    print(topic_choices)

    #topic_labels = ['Humble Life', 'Romance', 'Braggadocious', 'Music/Art', 'Partying', 'Smoking Weed']
    #get_topic_analysis(songs_df, topic_labels, topics_matrix, artist_name, album_list)
