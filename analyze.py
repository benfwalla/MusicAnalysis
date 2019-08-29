import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


def get_lexical_richness(songs_df: pd.DataFrame, artist_name, albums):
    songs_group = songs_df.groupby('Album')

    graph_df = pd.DataFrame(columns=('Album', 'Lexical Richness'))

    i = 0
    for name, album in songs_group:

        every_word_in_album = []

        for lyric in album['Lyrics'].iteritems():

            if isinstance(lyric[1], str):
                words = lyric[1].replace('\n', ' ')
                words = words.split(' ')

                filtered_words = [word for word in words if word not in stopwords.words('english') and len(word) > 1 and
                                  word not in ['na', 'la']]  # remove the stopwords

                every_word_in_album.extend(filtered_words)

        a = len(set(every_word_in_album))
        b = len(every_word_in_album)

        graph_df.loc[i] = (name, (a / float(b)) * 100)
        i += 1

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

    plt.show()


def get_sentiment_analysis(songs_df: pd.DataFrame, artist_name, albums):
    songs_group = songs_df.groupby('Album')

    graph_df = pd.DataFrame(columns=('Album', 'Positive', 'Neutral', 'Negative'))
    sid = SentimentIntensityAnalyzer()
    i = 0

    for name, album in songs_group:
        num_positive = 0
        num_negative = 0
        num_neutral = 0

        for lyric in album['Lyrics'].iteritems():

            if isinstance(lyric[1], str):

                sentences = lyric[1].split('\n')

                for sentence in sentences:
                    comp = sid.polarity_scores(sentence)
                    comp = comp['compound']
                    if comp >= 0.5:
                        num_positive += 1
                    elif -0.5 < comp < 0.5:
                        num_neutral += 1
                    else:
                        num_negative += 1

        num_total = num_negative + num_neutral + num_positive
        percent_negative = (num_negative / float(num_total)) * 100
        percent_neutral = (num_neutral / float(num_total)) * 100
        percent_positive = (num_positive / float(num_total)) * 100
        graph_df.loc[i] = (name, percent_positive, percent_neutral, percent_negative)
        i += 1

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

    plt.show()


def get_topic_choices(songs_df: pd.DataFrame, more_stop_words):

    stop_words = stopwords.words('english')
    stop_words.extend(more_stop_words)

    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=0.1)

    songs_df['Lyrics'] = songs_df['Lyrics'].str.replace('\r\r\n', ' ').str.replace('\r\r', ' ').str.replace('\n', ' ')
    songs_df['Lyrics'] = songs_df['Lyrics'].fillna('')

    tfidf = vectorizer.fit_transform(songs_df['Lyrics'])

    nmf = NMF(n_components=6)
    topics = nmf.fit_transform(tfidf)

    topic_choices = []
    for topic_num, topic in enumerate(nmf.components_):
        message = 'Topic #{}: '.format(topic_num + 1)
        message += ' '.join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-21 :-1]])
        topic_choices.append(message)

    return topic_choices, topics


def get_topic_analysis(songs_df, topic_labels, topics, artist_name, albums):

    df_topics = pd.DataFrame(topics, columns=topic_labels)

    for col in df_topics.columns:
        df_topics.loc[df_topics[col] >= 0.1, col] = 1
        df_topics.loc[df_topics[col] < 0.1, col] = 0

    songs_df = songs_df.join(df_topics)

    del songs_df['InAnAlbum']

    album_topics = songs_df.groupby('Album', sort=False).sum()
    album_topics = album_topics.reindex(albums)
    album_topics = album_topics.reset_index()

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

    plt.show()
