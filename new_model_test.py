import numpy as np
import pandas as pd
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

#create instance of fileManager
ml = fileManager();

#load new data
data = ml.loadIt("./Data/balancedData.pickle")


with open('Data/firstnames.txt') as f:
    firstnames = f.read().splitlines()

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.Plot, data.Genre, test_size=0.2, random_state=0)

#create list of stop words
stop_words = ENGLISH_STOP_WORDS.union(firstnames)

# make the pipeline
text_pipe = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words, lowercase=True)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])


#fit the model on the training data
text_pipe.fit(X_train, y_train);


# Prediction test
print(text_pipe.predict(X_test))

# Evaluate the performance on test set
grid_params = {
    'vect__stop_words': [None, 'english'],
    'tfidf__use_idf': (True, False),
}
search = GridSearchCV(text_pipe, grid_params)
search.fit(X_train, y_train);
print("Best parameters: ", search.best_params_)
print("Score: ",search.score(X_test, y_test))
print("Error Score: ", search.error_score)
print("Best estimator: ", search.best_estimator_)
print("Best Score: ", search.best_score_)


#false prediction
text = [
    "Gang kill police killed father man men brother family kills",
]

print("test prediction: ", text_pipe.predict(text))

#compress and serialize the model for the api
#ml.zipIt(text_pipe,"./Models/v4")





        