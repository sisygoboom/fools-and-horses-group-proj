from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier#, LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy
import warnings
warnings.filterwarnings('ignore')
from modelLoader import fileManager

class Model:
    def __init__(self, dataset_path="./Data/balancedData.pickle", model_path=None, stop_words=False, exclude_fnames=True, test_size=0.2):
        self.ml = fileManager()
        
        if model_path != None:
            self.text_pipe = self.ml.loadIt(model_path)
            
        self.data = self.ml.loadIt(dataset_path)
        # separate training and testing data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.Plot, 
            self.data.Genre, 
            test_size=test_size, 
            random_state=0)
        
        # define stopwords
        self.stop_words = None
        if stop_words:
            self.stop_words = ENGLISH_STOP_WORDS
            if exclude_fnames:
                with open('Data/firstnames.txt') as f:
                    firstnames = f.read().splitlines()
                self.stop_words.union(firstnames)
        
    def train(self):
        
        # instantiate text processing pipline
        self.text_pipe = Pipeline([
            ('vect', TfidfVectorizer(stop_words=self.stop_words, lowercase=True)),
            ('tfidf', TfidfTransformer(use_idf=True)),
            ('clf', SGDClassifier(loss='modified_huber')), # accuracy at last test: 78%
            #('clf', KNeighborsClassifier()),              # accuracy at last test: 46%
            #('clf', RandomForestClassifier()),            # accuracy at last test: 62%
            #('clf', MultinomialNB()),                     # accuracy at last test: 54%
            #('clf', LogisticRegression()),                  # accuracy at last test: 70%
        ])
        
        #fit model with training data
        self.text_pipe.fit(self.X_train, self.y_train)
        
    def get_pipe(self):
        return self.text_pipe
    
    def test(self):
        # Evaluate the performance on test set
        grid_params = {
            'vect__stop_words': [None, 'english', self.stop_words],
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
        
    def predict_custom(self, text):
        return self.text_pipe.predict([text])
    
    def predict_probability(self, text):
        return self.text_pipe.predict_proba([text])
    
    def get_genres(self):
         return self.text_pipe.named_steps['clf'].classes_
     
    def predict_plus_accuracy(self, text):
        probas = self.predict_probability(text)
        genres = self.get_genres()
        max_index = numpy.argmax(probas)
        return genres[max_index], probas[0][max_index]
        
    
    def predict_test_data(self):
        return self.text_pipe.predict(self.X_test)
        
        


if __name__ == "__main__":
    model = Model()
    model.train()
    #model.test()
    model.ml.zipIt(model.text_pipe, './Models/version4')
    
    # example
    #model = Model("./Data/example.pickle", False)
    #model.train(0.5)
    #model.ml.zipIt(model.get_pipe(), './Models/v6')
    