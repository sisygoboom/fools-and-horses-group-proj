import pandas as pd

class Dataset:
    def load_stored(self):
        data = pd.read_pickle('Data/dataset.pickle')
        return data
    
    def store_fresh(self):
        data = self.fresh_load()
        data.to_pickle('Data/dataset.pickle')
    
    def fresh_load(self):
        self.data = pd.read_csv("Data/wiki_movie_plots_deduped.csv", delimiter=',')
        self.sanitize()
        popularity = self.calculate_genre_popularity()
        self.fit_to_common_genre(popularity)
        return self.data
    
    def sanitize(self):
        self.data.Genre = [i.replace('-', ',')
                      .replace('/',',')
                      .replace(' ',',')
                      .strip('(')
                      .strip(')')
                      .strip(';')
                      .split(',') for i in self.data.Genre]
        
    def calculate_genre_popularity(self):
        popularity = {}
        for movie in self.data.Genre:
            for genre in movie:
                if(genre in  popularity.keys()):
                    popularity[genre] += 1
                else:
                    popularity[genre] = 1
        return popularity

    def fit_to_common_genre(self, popularity):
        for i, genres in enumerate(self.data.Genre):
            print(i, genres)
            highest = [genres[0], popularity[genres[0]]]
            for genre in genres:
                if highest[1] < popularity[genre]:
                    highest = [genre, popularity[genre]]
            print(highest)
            self.data.Genre[i] = highest[0]