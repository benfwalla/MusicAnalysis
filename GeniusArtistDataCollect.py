import os
import re
import requests
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup


class GeniusArtistDataCollect:
    """A wrapper class that is able to retrieve, clean, and organize all the album songs of a given artist
    Uses the Genius API and webscraping techniques to get the data."""

    def __init__(self, client_access_token, artist_name, albums):
        """
        Instantiate a GeniusArtistDataCollect object
        :param client_access_token: str - Token to access the Genius API. Create one at https://genius.com/developers
        :param artist_name: str - The name of the artist of interest
        :param albums: list - A list of all the artist's albums to be collected
        """

        self.client_access_token = client_access_token

        self.artist_name = artist_name

        self.albums = albums

        self.base_url = 'https://api.genius.com/'

        self.headers = {'Authorization': 'Bearer ' + self.client_access_token}

        self.artist_songs = None

    def search(self, query):
        """Makes a search request in the Genius API based on the query parameter. Returns a JSON response."""

        request_url = self.base_url + 'search'
        data = {'q': query}
        response = requests.get(request_url, data=data, headers=self.headers).json()

        return response

    def get_artist_songs(self):
        """Gets the songs of self.artist_name and places in a pandas.DataFrame"""

        # Search for the artist and get their id
        search_artist = self.search(self.artist_name)
        artist_id = str(search_artist['response']['hits'][0]['result']['primary_artist']['id'])

        print("ID: " + artist_id)

        # Initialize DataFrame
        df = pd.DataFrame(columns=['Title', 'URL'])

        # Iterate through all the pages of the artist's songs
        more_pages = True
        page = 1
        i = 0
        while more_pages:

            print("page: " + str(page))

            # Make a request to get the songs of an artist on a given page
            request_url = self.base_url + 'artists/' + artist_id + '/songs' + '?per_page=50&page=' + str(page)
            response = requests.get(request_url, headers=self.headers).json()

            print(response)

            # For each song which the given artist is the primary_artist of the song, add the song title and
            # Genius URL to the DataFrame
            for song in response['response']['songs']:

                if str(song['primary_artist']['id']) == artist_id:

                    title = song['title']
                    url = song['url']

                    df.loc[i] = [title, url]
                    i += 1

            page += 1

            if response['response']['next_page'] is None:
                more_pages = False

        # Get the HTML, Album Name, and Song Lyrics from helper methods in the class
        df['html'] = df['URL'].apply(self.get_song_html)
        df['Album'] = df['html'].apply(self.get_album_from_html)
        df['InAnAlbum'] = df['Album'].apply(lambda a: self.is_track_in_an_album(a, self.albums))
        df = df[df['InAnAlbum'] == True]
        df['Lyrics'] = df.apply(lambda row: self.get_lyrics(row.html), axis=1)

        del df['html']

        self.artist_songs = df

        return self.artist_songs

    def get_song_html(self, url):
        """Scrapes the entire HTML of the url parameter"""

        request = urllib.request.Request(url)
        request.add_header("Authorization", "Bearer " + self.client_access_token)
        request.add_header("User-Agent",
                           "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")
        page = urllib.request.urlopen(request)
        html = BeautifulSoup(page, "lxml")

        print("Scraped: " + url)
        return html

    def get_lyrics(self, html):
        """Scrapes the html parameter to get the song lyrics on a Genius page in one, large String object"""

        lyrics = html.find("div", class_="lyrics")

        all_words = ''

        # Clean lyrics
        for line in lyrics.get_text():
            all_words += line

        # Remove identifiers like chorus, verse, etc
        all_words = re.sub(r'[\(\[].*?[\)\]]', '', all_words)

        # remove empty lines, extra spaces, and special characters
        all_words = os.linesep.join([s for s in all_words.splitlines() if s])
        all_words = all_words.replace('\r', '')
        all_words = all_words.replace('  ', ' ')

        return all_words

    def get_album_from_html(self, html):
        """Scrapes the html parameter to get the album name of the song on a Genius page"""

        parse = html.findAll("span")

        album = ''

        for i in range(len(parse)):
            if parse[i].text == 'Album':
                i += 1
                album = parse[i].text.strip()
                break

        return album

    def is_track_in_an_album(self, album, albums_list):
        """A helper method to see if a given album (which would be tied with a song) is in the given album_list"""

        if album in albums_list:
            return True
        else:
            return False
