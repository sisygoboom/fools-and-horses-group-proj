from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from modelLoader import fileManager

class Model:
    def __init__(self, dataset_path="./Data/balancedData.pickle", exclude_fnames=True):
        self.ml = fileManager()
        self.data = self.ml.loadIt(dataset_path)
        
        # define stopwords
        self.stop_words = ENGLISH_STOP_WORDS
        if exclude_fnames:
            with open('Data/firstnames.txt') as f:
                firstnames = f.read().splitlines()
            self.stop_words.union(firstnames)
        
    def train(self, test_size=0.2):
        # separate training and testing data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.Plot, 
            self.data.Genre, 
            test_size=test_size, 
            random_state=0)
        
        # instantiate text processing pipline
        self.text_pipe = Pipeline([
            ('vect', TfidfVectorizer(stop_words=self.stop_words, lowercase=True)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        
        #fit model with training data
        self.text_pipe.fit(self.X_train, self.y_train)
        
    def get_pipe(self):
        return self.text_pipe
    
    def test(self):
        # Prediction test
        print(self.text_pipe.predict(self.X_test))
        
        # Evaluate the performance on test set
        grid_params = {
            'vect__stop_words': [None, 'english'],
            'tfidf__use_idf': (True, False),
        }
        search = GridSearchCV(self.text_pipe, grid_params)
        search.fit(self.X_train, self.y_train)
        
        return {
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'error_score': search.error_score,
            'score': search.score(self.X_test, self.y_test),
            'best_parameters': search.best_params_
        }
        
        
        


if __name__ == "__main__":
    model = Model()
    model.train()
    #model.test()
    model.ml.zipIt(model.text_pipe, './Models/version4')
    
    # example
    #model = Model("./Data/example.pickle", False)
    #model.train(0.5)
    #model.ml.zipIt(model.get_pipe(), './Models/v6')
    