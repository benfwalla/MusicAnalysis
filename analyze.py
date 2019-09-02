import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def get_lexical_richness(songs_df: pd.DataFrame, artist_name, albums):
    """Plots the lexical richness of each album according to the lyrics in the given DataFrame"""

    songs_group = songs_df.groupby('Album')

    graph_df = pd.DataFrame(columns=('Album', 'Lexical Richness'))

    i = 0
    for name, album in songs_group:

        # Create a list of words for every word that is in an album
        every_word_in_album = []

        for lyric in album['Lyrics'].iteritems():

            if isinstance(lyric[1], str):
                words = lyric[1].replace('\n', ' ')
                words = words.split(' ')

                filtered_words = [word for word in words if word not in stopwords.words('english') and len(word) > 1 and
                                  word not in ['na', 'la']]  # remove the stopwords

                every_word_in_album.extend(filtered_words)

        # Find how many words and unique words are in the album
        a = len(set(every_word_in_album))
        b = len(every_word_in_album)

        # Calculate the lexical richness of the album, which is the amount of unique words relative to
        # the total amount of words in the album
        graph_df.loc[i] = (name, (a / float(b)) * 100)
        i += 1

    # Plot the lexical richness by album on a bar chart
    graph_df = graph_df.set_index('Album')
    graph_df = graph_df.reindex(albums)
    graph_df.plot(kind='bar',
                  x=graph_df.index.values,
                  y='Lexical Richness',
                  title='Lexical richness of each {} Album'.format(artist_name),
                  legend=None)

    plt.xlabel("Albums")
    plt.xticks(rotation=85)
    plt.tight_layout()
    plt.savefig('{} Lexical Richness.png'.format(artist_name))

    # Show the bar chart
    plt.show()


def get_sentiment_analysis(songs_df: pd.DataFrame, artist_name, albums):
    """Plots the sentiment analysis of each album according to the song lyrics in the given DataFrame.
    Sentiment Analysis shows how much of the content is categorized in positive, negative, or neutral."""

    songs_group = songs_df.groupby('Album')

    # Create a resulting DataFrame
    graph_df = pd.DataFrame(columns=('Album', 'Positive', 'Neutral', 'Negative'))

    # Initialize a SentimentIntensityAnalyzer object from the nltk library. This is what does the magic!
    sid = SentimentIntensityAnalyzer()

    i = 0
    for name, album in songs_group:
        num_positive = 0
        num_negative = 0
        num_neutral = 0

        for lyric in album['Lyrics'].iteritems():

            if isinstance(lyric[1], str):

                # Split up all the lyrics by sentences (or the lines in a song)
                sentences = lyric[1].split('\n')

                for sentence in sentences:
                    # Determine the polarity score of each sentence. To put simply, this will quantify how happy
                    # or song a sentence is
                    comp = sid.polarity_scores(sentence)
                    comp = comp['compound']

                    # Based on the polarity score, categorize where this sentence should fall
                    if comp >= 0.5:
                        num_positive += 1
                    elif -0.5 < comp < 0.5:
                        num_neutral += 1
                    else:
                        num_negative += 1

        # Compute the percentage of negative, neutral, and positive sentences relative to the total amount of sentences
        # in the album
        num_total = num_negative + num_neutral + num_positive
        percent_negative = (num_negative / float(num_total)) * 100
        percent_neutral = (num_neutral / float(num_total)) * 100
        percent_positive = (num_positive / float(num_total)) * 100
        graph_df.loc[i] = (name, percent_positive, percent_neutral, percent_negative)
        i += 1

    # Graph the results on a bar chart
    graph_df = graph_df.set_index('Album')
    graph_df = graph_df.reindex(albums)
    graph_df.plot(kind='bar',
                  x=graph_df.index.values,
                  title='Sentiment Analysis of each {} Album'.format(artist_name),
                  stacked=True,
                  color=['green', 'orange', 'red'])
    plt.xlabel("Albums")
    plt.xticks(rotation=85)
    plt.tight_layout()
    plt.savefig('{} Sentiment Analysis.png'.format(artist_name))

    # Show the bar chart
    plt.show()


def get_topic_choices(songs_df: pd.DataFrame, more_stop_words):
    """A helper method for get_topic_analysis() that provides the first step in the topic analysis. This function
    return the top 20 words in each topic as well as the numeric matrix of the topics. It is then up to the
    user to inspect those 20 words and determine a one or two-word topic that fits those words."""

    # Get a list of all the deplorable stop words in the english language
    stop_words = stopwords.words('english')
    stop_words.extend(more_stop_words)

    # Create a vector based on term frequency-inverse document frequency
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=0.1)

    # Clean the lyric data
    songs_df['Lyrics'] = songs_df['Lyrics'].str.replace('\r\r\n', ' ').str.replace('\r\r', ' ').str.replace('\n', ' ')
    songs_df['Lyrics'] = songs_df['Lyrics'].fillna('')

    tfidf = vectorizer.fit_transform(songs_df['Lyrics'])

    # Perform an NMF, or non-negative matrix factorization, to pull out co-occurring word groupings that are found
    # in the discography. We'll get the top six topic groupings.
    nmf = NMF(n_components=6)
    topics_matrix = nmf.fit_transform(tfidf)

    # Store the six topic groups into a list for the user to view and make a decision on the topic name
    topic_choices = []
    for topic_num, topic in enumerate(nmf.components_):
        message = 'Topic #{}: '.format(topic_num + 1)
        message += ' '.join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-21 :-1]])
        topic_choices.append(message)

    return topic_choices, topics_matrix


def get_topic_analysis(songs_df, topic_labels, topics_matrix, artist_name, albums):
    """Plots the topic analysis of each album based on the topics provided. This allows us to see how an artists
    career has evolved based on the six topics provided."""

    # Create a DataFrame based on the topics_matrix values with topic_labels as the column headers. The matrix
    # can be created with the get_topic_choices() helper function.
    df_topics = pd.DataFrame(topics_matrix, columns=topic_labels)

    # Place a boolean value on each song's matrix value of a topic to determine if the song mentions that topic or not.
    for col in df_topics.columns:
        df_topics.loc[df_topics[col] >= 0.1, col] = 1
        df_topics.loc[df_topics[col] < 0.1, col] = 0

    songs_df = songs_df.join(df_topics)

    del songs_df['InAnAlbum']

    # Group all the songs by their albums and sum up the results of topic relevance
    album_topics = songs_df.groupby('Album', sort=False).sum()
    album_topics = album_topics.reindex(albums)
    album_topics = album_topics.reset_index()

    # Plot the results on a line chart
    plt.figure(figsize=(20, 10))
    for col in album_topics.columns:
        if col != 'Album':
            plt.plot(album_topics['Album'], album_topics[col], label=col, linewidth=4.0)

    plt.xlabel("Albums")
    plt.xticks(rotation=85)
    plt.grid(True)
    plt.tight_layout()

    plt.legend()

    plt.savefig('{} Topic Analysis.png'.format(artist_name))

    # Show the line chart
    plt.show()
